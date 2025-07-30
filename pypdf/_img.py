import datetime
import math
from io import BytesIO
from typing import cast

from PIL import Image, ImageOps

from ._color import ColorSpaceWrapper
from .constants import ColorSpaces, FilterTypes, ImageAttributes, StreamAttributes
from .errors import PyPdfError
from .generic import ArrayObject, IndirectObject, NameObject, StreamObject


class BitsToIntProcessor:
    """
    §8.9.3 Sample representation of ISO 32000-2:2017
    Returnes the color values of a byte stream as ints according the BitsPerComponent entry
    in the Stream dictionary.
    According PDF spec image data byte stream has to interpreted as bit stream and values
    of the pixels must be computed for BitsPerComponent != 8
    """

    def __init__(self, bits_per_component: int) -> None:
        if bits_per_component > 16:
            raise PyPdfError(f"Unsupported BitsPerComponent value: {bits_per_component!s}")
        self.bits_per_component: int = bits_per_component

    def process(self, input: bytes) -> list[int]:
        if self.bits_per_component == 8:
            return list(bytearray(input))

        output: list[int] = []
        bit_stream = BytesIO(input)
        b = int.from_bytes(bit_stream.read(1))
        if b == b"":
            raise ValueError("Input data are empty.")
        loaded_bits = 8
        sample: int = 0

        while loaded_bits > 0:
            if loaded_bits >= self.bits_per_component:
                sample = b >> min(loaded_bits - self.bits_per_component, loaded_bits)
                loaded_bits -= self.bits_per_component
                b = (b & int(math.exp2(loaded_bits) - 1))
                output.append(sample)
            while loaded_bits < self.bits_per_component:
                r = bit_stream.read(1)
                if r == b"":
                    if loaded_bits > 0:
                        fill_with_zeros = self.bits_per_component - loaded_bits
                        b = b << fill_with_zeros
                        loaded_bits += fill_with_zeros
                    break
                loaded_bits += 8
                b = (b << 8) | int.from_bytes(r)

        return output


class DecodingProcessor:
    """
    §8.9.3 Sample representation of ISO 32000-2:2017
    Color values are represented as float according PDF spec.
    Returnes the color values in the range of the Decode array (typical [0.0, 1.0])
    that is given in the Stream dictionary of an XObject image explicitely or
    applies implicite by the default array for the given color space.
    """

    def __init__(self, decode_arr: ArrayObject|list[float]) -> None:
        self.decode_arr: list[float] = list(decode_arr)

    def decode_to_floats(self, input: list[int], bits_per_component: int) -> list[float]:
        output: list[float] = []
        number_of_components = int(len(self.decode_arr) / 2)

        # precalculates values to increase performace
        d = (math.exp2(bits_per_component) - 1)
        precalculated_values: list[tuple[float, ...]] = []
        for component_index in range(number_of_components):
            mi = self.decode_arr[component_index * 2]
            ma = self.decode_arr[component_index * 2 + 1]
            precalculated_values.append((mi, (ma - mi) / d))

        index = 0
        length = len(input)
        while index < length:
            for factor in precalculated_values:
                 # factor[0] == lower bound, factor[1] == the actual factor to multiply with
                y = factor[0] + input[index] * factor[1]
                output.append(y)
                index += 1

        return output

    def decode_to_bytes(self, input: list[int], bits_per_component: int) -> bytes:
        output_floats = self.decode_to_floats(input, bits_per_component)
        return bytes([int(f*255) for f in output_floats])


class XObjectWrapper:

    def __init__(self, xobj: IndirectObject|StreamObject) -> None:
        self.img_obj: StreamObject = xobj.get_object() # type: ignore
        if self.img_obj.get(ImageAttributes.SUBTYPE) != "/Image":
            raise PyPdfError("Not a XObject image stream.")

        self.colorspace: ColorSpaceWrapper = ColorSpaceWrapper.from_xobj(self.img_obj)
        self.filters: NameObject|ArrayObject|None = self.img_obj.get(StreamAttributes.FILTER)
        self.width: int = cast(int, self.img_obj.get(ImageAttributes.WIDTH))
        self.height: int = cast(int, self.img_obj.get(ImageAttributes.HEIGHT))

    def _is_image_mask(self, xobj: StreamObject) -> bool:
        return xobj.get(ImageAttributes.IMAGE_MASK, False)

    def _requires_bit_stream_conversion(self) -> bool:
        raise PyPdfError("Not yet implemented")

    def get_color_space(self) -> ColorSpaceWrapper:
        return self.colorspace

    def get_image_data(self) -> bytes:
        return self.img_obj.get_data()

    def get_PIL_image(self, use_alternate_colorspace: bool = True) -> Image.Image|None:
        PIL_mode = {
            ColorSpaces.DEVICE_GRAY: "L",
            ColorSpaces.DEVICE_RGB: "RGB",
            ColorSpaces.DEVICE_CMYK: "CMYK",
            ColorSpaces.LAB: "LAB"
        }

        PIL_img = None

        # first check the filter/encoding and get image data as bytes
        img_colorspace_name = self.colorspace.get_colorspace_type()
        img_data: bytes = b""

        if self.filters in [FilterTypes.DCT_DECODE, FilterTypes.JPX_DECODE]:
            # Her we can optimize depending on the color space, and just return the JPEG as is ...
            if img_colorspace_name in [ColorSpaces.DEVICE_GRAY,
                                  ColorSpaces.DEVICE_RGB,
                                  ] and not self._is_image_mask(self.img_obj):
                PIL_img = Image.open(BytesIO(self.img_obj.get_data()))
                if PIL_img.mode == "YCbCr":
                    PIL_img = PIL_img.convert("RGB")
                if PIL_img.mode == "L":
                    PIL_img = ImageOps.invert(PIL_img)
                return PIL_img

            # in case of a JPEG we need the image data bytes from PIL to convert to alternate color space
            #t1 = datetime.datetime.now()
            PIL_img = Image.open(BytesIO(self.img_obj.get_data()))
            pil_img_data = list(PIL_img.getdata())
            bands_count = len(PIL_img.getbands())
            if bands_count > 1:
                temp = []
                for band in pil_img_data:
                    temp.extend(band)
                pil_img_data = temp
            img_data = bytes(pil_img_data)
            #t2 = datetime.datetime.now()
            #print(f"Img bytes from PIL: {t2-t1}")

        else:
            # simplified for testing for none-JPEG data
            # depending on a predictor for Flate/LZW the BitsToIntProcessor becomes obsolete later
            # this is not handled at the moment
            img_data = self.img_obj.get_data()

        # now we do the color processing for the image data bytes depending on the actual color space type
        if img_colorspace_name in [ColorSpaces.DEVICE_N] and not self._is_image_mask(self.img_obj):
            if use_alternate_colorspace:
                bits_per_component = self.img_obj.get("/BitsPerComponent", 8)
                number_of_colorants = self.colorspace.get_colorant_count()

                #t1 = datetime.datetime.now()
                data_as_int = BitsToIntProcessor(bits_per_component).process(img_data)
                #t2 = datetime.datetime.now()
                #print(f"Data as Int {t2-t1}")

                default_decode_array = self.colorspace.get_default_decode_array()

                #t1 = datetime.datetime.now()
                data_as_float = DecodingProcessor(self.img_obj.get("/Decode", default_decode_array)).decode_to_floats(data_as_int, bits_per_component)
                #t2 = datetime.datetime.now()
                #print(f"Decoding {t2-t1}")

                #t1 = datetime.datetime.now()
                alternate_colorspace_data_as_bytes = self.colorspace.get_tint_transform_object().process_image_data_to_bytes(data_as_float, number_of_colorants) # type: ignore
                #t2 = datetime.datetime.now()
                #print(f"Tint Transform {t2-t1}")

                alternate_colorspace_name = self.colorspace.get_alternate_colorspace_name()

                PIL_img = Image.frombytes(
                    PIL_mode.get(alternate_colorspace_name), # type: ignore
                    (self.width, self.height),
                    alternate_colorspace_data_as_bytes)
            else:
                raise PyPdfError("Creation of multi-channel images not yet implemented for DeviceN color space.")

        elif img_colorspace_name in [ColorSpaces.DEVICE_CMYK] and not self._is_image_mask(self.img_obj):
            bits_per_component = self.img_obj.get("/BitsPerComponent", 8)
            data_as_int = BitsToIntProcessor(bits_per_component).process(img_data)

            # JPEG image data are inverted for CMYK
            inverted = self.filters in [FilterTypes.DCT_DECODE, FilterTypes.JPX_DECODE]

            # optimization
            if (bits_per_component == 8) and (self.img_obj.get("/Decode") is None) and not inverted:
                data_as_bytes = bytes(data_as_int)
            else:
                default_decode_array = self.colorspace.get_default_decode_array(inverted=inverted)
                decode_array = self.img_obj.get("/Decode", default_decode_array)

                data_as_bytes = DecodingProcessor(decode_array).decode_to_bytes(data_as_int, bits_per_component)

            PIL_img = Image.frombytes(
                PIL_mode.get(self.colorspace.get_colorspace_type()), # type: ignore
                (self.width, self.height),
                data_as_bytes)
        else:
            raise PyPdfError(f"Color space {self.colorspace.cs_obj} not yet implemented")

        return PIL_img





