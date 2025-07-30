import math
from typing import Iterable, cast

from ._utils import logger_warning
from .constants import ColorSpaces
from .errors import PyPdfError
from .generic import (
    ArrayObject,
    DictionaryObject,
    IndirectObject,
    NameObject,
    StreamObject,
)


class PostScriptProcessor:

    def __init__(self, ps_code: str) -> None:
        self.ps_code: str = ps_code
        self.stack: list[float|str|bool|list] = []

    def tokenize(self, code:str) -> list[str]:
        """Splits Postscript code into a token list to be parsed and interpreted later"""
        return code.replace("{", " { ").replace("}", " } ").split()

    def parse_procedure(self, iterator: Iterable) -> list[str]:
        """
        Returnes the code (tokens) of a procedure and possible contained
        sub-procedures in a token list
        """
        procedure = []
        stack = 1  # topmost level ("{")
        for token in iterator:
            if token == "{":
                stack += 1
            elif token == "}":
                stack -= 1
                if stack == 0:  # topmost procedure is finished
                    return procedure
            procedure.append(token)
        raise ValueError("Unmatched opening brace '{' detected.")

    def execute(self, token: str) -> None:  # noqa: C901, PLR0912, PLR0915
        """
        Interpretes and executes a single PostScript operator

        According PostScript language reference manual / Adobe Systems Incorporated. — 3rd ed.
        (https://www.adobe.com/jp/print/postscript/pdfs/PLRM.pdf)
        """
        try:
            self.stack.append(float(token))  # put number on the stack
        except ValueError:
            # Math operators
            if token == "add":
                self.ensure_stack_size(2, token)
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a + b)
            elif token == "sub":
                self.ensure_stack_size(2, token)
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a - b)
            elif token == "mul":
                self.ensure_stack_size(2, token)
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a * b)
            elif token == "div":
                self.ensure_stack_size(2, token)
                b = self.stack.pop()
                a = self.stack.pop()
                if b == 0:
                    raise ZeroDivisionError("Division by Zero not allowed.")
                self.stack.append(a / b)
            elif token == "cvr":
                self.ensure_stack_size(1, token)
                value = self.stack.pop()
                self.stack.append(float(value))
            elif token == "abs":
                self.ensure_stack_size(1, token)
                self.stack.append(abs(self.stack.pop()))
            elif token == "cvi":
                self.ensure_stack_size(1, token)
                self.stack.append(int(self.stack.pop()))
            elif token == "floor":
                self.ensure_stack_size(1, token)
                self.stack.append(math.floor(self.stack.pop()))
            elif token == "mod":
                self.ensure_stack_size(2, token)
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a % b)
            elif token == "sin":
                self.ensure_stack_size(1, token)
                self.stack.append(math.sin(math.radians(self.stack.pop())))
            elif token == "sqrt":
                self.ensure_stack_size(1, token)
                value = self.stack.pop()
                if value < 0:
                    raise ValueError("Cannot calculate square root of a negative number.")
                self.stack.append(math.sqrt(value))
            elif token == "atan":
                self.ensure_stack_size(2, token)
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(math.degrees(math.atan2(a, b)))
            elif token == "idiv":
                self.ensure_stack_size(2, token)
                b = self.stack.pop()
                a = self.stack.pop()
                if b == 0:
                    raise ZeroDivisionError("Division by zero in idiv.")
                self.stack.append(a // b)
            elif token == "ln":
                self.ensure_stack_size(1, token)
                value = self.stack.pop()
                if value <= 0:
                    raise ValueError("Logarithm not defined for non-positive values.")
                self.stack.append(math.log(value))
            elif token == "neg":
                self.ensure_stack_size(1, token)
                self.stack.append(-self.stack.pop())
            elif token == "ceiling":
                self.ensure_stack_size(1, token)
                self.stack.append(math.ceil(self.stack.pop()))
            elif token == "exp":
                self.ensure_stack_size(1, token)
                self.stack.append(math.exp(self.stack.pop()))
            elif token == "log":
                self.ensure_stack_size(1, token)
                value = self.stack.pop()
                if value <= 0:
                    raise ValueError("Logarithm not defined for non-positive values.")
                self.stack.append(math.log10(value))
            elif token == "round":
                self.ensure_stack_size(1, token)
                self.stack.append(round(self.stack.pop()))
            elif token == "truncate":
                self.ensure_stack_size(1, token)
                self.stack.append(math.trunc(self.stack.pop()))
            elif token == "cos":
                self.ensure_stack_size(1, token)
                self.stack.append(math.cos(math.radians(self.stack.pop())))
            # Logical and bitwise operators
            elif token == "true":
                self.stack.append(True)
            elif token == "false":
                self.stack.append(False)
            elif token == "eq":
                self.ensure_stack_size(2, token)
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a = b)
            elif token == "ne":
                self.ensure_stack_size(2, token)
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a != b)
            elif token == "lt":
                self.ensure_stack_size(2, token)
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a < b)
            elif token == "le":
                self.ensure_stack_size(2, token)
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a <= b)
            elif token == "gt":
                self.ensure_stack_size(2, token)
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a > b)
            elif token == "ge":
                self.ensure_stack_size(2, token)
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a >= b)
            # todo: add "and, or, xor, or" bitwise and logical
            # todo: add bitshift
            # Stack operations
            elif token == "dup":
                self.ensure_stack_size(1, token)
                self.stack.append(self.stack[-1])
            elif token == "roll":
                self.ensure_stack_size(2, token)
                m = int(self.stack.pop())
                n = int(self.stack.pop())
                if len(self.stack) < n:
                    raise ValueError(f"Not enough elements on stack for roll: required {n}, found {len(self.stack)}")
                m %= n
                if m > 0:
                    self.stack[-n:] = self.stack[-m:] + self.stack[-n:-m]
                elif m < 0:
                    m = abs(m)
                    self.stack[-n:] = self.stack[-n+m:] + self.stack[-n:-n+m]
            elif token == "index":
                self.ensure_stack_size(1, token)
                index = int(self.stack.pop())
                if index < 0 or index >= len(self.stack):
                    raise ValueError(f"Index out of range: {index}")
                self.stack.append(self.stack[-1 - index])
            elif token == "exch":
                self.ensure_stack_size(2, token)
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(b)
                self.stack.append(a)
            elif token == "pop":
                self.ensure_stack_size(1, token)
                self.stack.pop()
            elif token == "copy":
                self.ensure_stack_size(1, token)
                n = int(self.stack.pop())
                if len(self.stack) < n:
                    raise ValueError(f"Not enough elements on stack for copy: required {n}, found {len(self.stack)}")
                self.stack.extend(self.stack[-n:])
            # Conditional operators
            elif token == "if":
                self.ensure_stack_size(2, token)
                proc = self.stack.pop()
                condition = self.stack.pop()
                if isinstance(proc, list):  # procedures are on the stack and must be of type list
                    if condition:
                        self.execute_tokens(proc)
                else:
                    raise ValueError(f"Expected a procedure, but got: {proc}")
            elif token == "ifelse":
                self.ensure_stack_size(3, token)
                proc2 = self.stack.pop()
                proc1 = self.stack.pop()
                condition = self.stack.pop()
                if isinstance(proc1, list) and isinstance(proc2, list):
                    if condition:
                        self.execute_tokens(proc1)
                    else:
                        self.execute_tokens(proc2)
                else:
                    raise ValueError(f"Expected two procedures, but got: {proc1} and {proc2}")
            else:
                raise ValueError(f"Unknown operator or invalid value: {token}")

    def ensure_stack_size(self, size: int, operator: str) -> None:
        """
        Ensures that are at least `size` elements on the stack.
        Returnes an error message with the actual operator if not.
        """
        if len(self.stack) < size:
            raise ValueError(f"Operator '{operator}' requires at least {size} stack elements, but only {len(self.stack)} present.")  # noqa: E501

    def execute_tokens(self, tokens: list[str]) -> None:
        """Executes a list of tokens (operators)."""
        iterator = iter(tokens)
        for token in iterator:
            if token == "{":
                # puts a procedure on the stack to be used by one of the next operators:
                procedure = self.parse_procedure(iterator)
                self.stack.append(procedure)
            else:
                self.execute(token)

    def process(self, inputs: list[float]) -> list[float]:
        """
        Executes the PostScript code with the given values in 'inputs'.

        Parameters:
            inputs (list[float]): Initial values to be put on the stack.

        Returns:
            list[float]: The values on the stack after the code has been executed.
        """
        self.stack.clear() # In case this instance is cached we must ensure that the stack is initially empty
        self.stack.extend(inputs)
        tokens = self.tokenize(self.ps_code)

        self.execute_tokens(tokens)
        if isinstance(self.stack[-1], list): # Code from PDF stream mostly is given as procedure enclosed in '{}'
            self.execute_tokens(cast(list, self.stack.pop()))

        return self.stack # type: ignore


class TintTransformWarpper:

    def __init__(self, func: DictionaryObject|StreamObject) -> None:

        function_types: list[int] = [0,2,3,4]

        self.func_obj: DictionaryObject|StreamObject = func.get_object() # type: ignore
        function_type = self.func_obj.get("/FunctionType", -1)
        if function_type not in function_types:
            raise ValueError("The given Dictionary or Stream is not a PDF function")

        self.domain = list(cast(ArrayObject, self.func_obj.get("/Domain")))
        self.rang = self.func_obj.get("/Range") # can be None, will be handled later
        self.tint_transform_function = getattr(self, f"_process_type_{function_type!s}")

        self.cache: dict = {}

    def _clip_to_range(self, boundaries: list[float], input: list[float]) -> None:
        """
        Applies the clipping, if necessary,
        according the Domain and Range arrays in the TintTransform array
        """
        for index in range(len(input)):
            lower_bound = boundaries[index * 2]
            upper_bound = boundaries[index * 2 + 1]
            input[index] = max(input[index], lower_bound)
            input[index] = min(input[index], upper_bound)

    def _process_type_0(self, input: list[float]) -> list[float]:
        """Sampled function"""
        raise PyPdfError("Function type 0 not yet implemented")

    def _process_type_2(self, input: list[float]) -> list[float]:
        """Exponential function"""
        raise PyPdfError("Function type 2 not yet implemented")

    def _process_type_3(self, input: list[float]) -> list[float]:
        """Stitching function"""
        raise PyPdfError("Function type 3 not yet implemented")

    def _process_type_4(self, input: list[float]) -> list[float]:
        """PostScript function"""
        ps_code = self.cache.get("ps_code")
        if not ps_code:
            ps_code = str(cast(StreamObject, self.func_obj).get_data().decode())
            self.cache.update({"ps_code":ps_code})
        processor = self.cache.get("processor")
        if not processor:
            processor = PostScriptProcessor(ps_code)
            self.cache.update({"processor":processor})
        return processor.process(input)

    def process_colorant_color_sequence(self, input: list[float]) -> list[float]:
        """Processes a single pixel, i.e. the color given as a squence of the colorant values"""
        self._clip_to_range(self.domain, input)
        outputs = self.tint_transform_function(input)
        if self.rang:
            self._clip_to_range(self.rang, outputs)
        return outputs

    def process_image_data(self, input: list[float], input_colorant_count: int) -> list[float]:
        """Processes the whole image bytes at once and returnes them in the alternate color space"""
        output: list[float] = []
        index = 0
        length = len(input)
        color_cache = {}

        while index < length:
            color_sequence = input[index:index+input_colorant_count]
            index += input_colorant_count
            cache_key = str(color_sequence) # Quick and dirty approach to use the input values as key for the cache
            result = color_cache.get(cache_key)
            if not result:
                result = self.process_colorant_color_sequence(color_sequence)
                color_cache.update({cache_key:result.copy()})
            output.extend(result)

        return output

    def process_image_data_to_bytes(self, input: list[float], input_colorant_count: int) -> bytes:
        """
        Processes the whole image bytes at once and returnes them in the alternate color space as bytes
        ready to use for the creation of a PIL image
        """
        return bytes([int(f * 255) for f in self.process_image_data(input, input_colorant_count)])


class ColorSpaceWrapper:
    """Referes to the §8.6 Colour Space of the actual ISO 32000-2:2017"""

    @classmethod
    def from_xobj(cls, obj: StreamObject|IndirectObject) -> "ColorSpaceWrapper":
        """Takes the ColorSpace from an XObject image Stream"""
        colorspace: ArrayObject|NameObject = obj.get_object().get("/ColorSpace")
        return ColorSpaceWrapper(colorspace)

    def __init__(self, colorspace: ArrayObject|NameObject) -> None:
        # Read the color space types from the constants
        self.cs_types = [getattr(ColorSpaces, cs) for cs in dir(ColorSpaces)
                            if not cs.startswith("__") and not callable(getattr(ColorSpaces, cs))]

        self.cs_obj: ArrayObject|NameObject = colorspace.get_object() # type: ignore
        self.cs_type: NameObject = self.cs_obj[0] if isinstance(self.cs_obj, ArrayObject) else self.cs_obj
        # Validate if we have a color space or somthing else
        if self.cs_type not in self.cs_types:
            raise ValueError("The PdfObject is not a color space array or name.")

    def get_colorspace_type(self) -> NameObject:
        return self.cs_type

    def get_colorant_names(self) -> list[str|NameObject]|ArrayObject:
        """
        Returns the explicitely defined colorants in case of a Separation or DeviceN color space
        or the implicite given colorants of the color spaces with fixed colorants
        """
        cs_type = self.cs_type

        if cs_type == ColorSpaces.INDEXED:
            # Here the colorants must be taken from the base color space
            base_colorspace = self.cs_obj[1].get_object() # type: ignore
            if isinstance(base_colorspace, NameObject):
                cs_type = base_colorspace # Fall through and return the colorants of one of the types below
            elif isinstance(base_colorspace, ArrayObject):
                return ColorSpaceWrapper(base_colorspace).get_colorant_names()

        if self.cs_type == ColorSpaces.ICCBASED:
            # Spec defines only Gray, RGB and CMYK ICC-Profiles
            # After getting the numnber of components we fall through and
            # return the colorants of one of the types below
            icc_profile_stream = cast(StreamObject, self.cs_obj[1].get_object()) # type: ignore
            number_of_components = int(str(icc_profile_stream.get("/N")))
            if number_of_components == 1:
                cs_type = ColorSpaces.DEVICE_GRAY
            elif number_of_components == 3:
                cs_type = ColorSpaces.DEVICE_RGB # very optimistic, could be a Lab profile but very, very rare case
            elif number_of_components == 4:
                cs_type = ColorSpaces.DEVICE_CMYK
            else:
                raise PyPdfError(f"Invalid number of components in ICCBased color space in {self.cs_obj!s}.")

        if cs_type == ColorSpaces.DEVICE_GRAY:
            return ["Gray"]

        if cs_type == ColorSpaces.DEVICE_RGB:
            return ["Red", "Green", "Blue"] # Fixed sequence order according PDF sepc

        if cs_type == ColorSpaces.DEVICE_CMYK:
            return ["Cyan", "Magenta", "Yellow", "Black"] # Fixed sequence order according PDF sepc

        if cs_type in [ColorSpaces.CAL_GRAY, ColorSpaces.CAL_RGB]:
            return ["X", "Y", "Z"] # Fixed sequence order according PDF sepc

        if cs_type == ColorSpaces.LAB:
            return ["L", "a", "b"] # Fixed sequence order according PDF sepc

        if cs_type in [ColorSpaces.SEPARATION]:
            return [self.cs_obj[1][1:]] # Omit the leading '/' of the name object

        if cs_type in [ColorSpaces.DEVICE_N]:
            return [colorant[1:] for colorant in list(self.cs_obj[1])] # Omit the leading '/' of the name object(s)

        if cs_type == ColorSpaces.PATTERN:
            logger_warning(
                "Parsing and interpreting of the pattern dictionary not yet implemented. No colorants returned",
                __name__,
            )
            return []

        raise PyPdfError(f"Determination of the colorants of the color space failed: {self.cs_obj!s}")

    def get_colorant_count(self) -> int:
        """
        Returns the number of colorants (channels) of a color space,
        required for color computations like the conversion to the
        alternate color space.
        """
        cs_type = self.cs_type

        if cs_type == ColorSpaces.INDEXED:
            base_colorspace = self.cs_obj[1].get_object() # type: ignore
            if isinstance(base_colorspace, NameObject):
                cs_type = base_colorspace # We read the base color space and fall through to get the values further down
            elif isinstance(base_colorspace, ArrayObject):
                return ColorSpaceWrapper(base_colorspace).get_colorant_count()

        if cs_type in [ColorSpaces.DEVICE_GRAY, ColorSpaces.SEPARATION]:
            return 1

        if cs_type in [ColorSpaces.CAL_GRAY, ColorSpaces.CAL_RGB, ColorSpaces.DEVICE_RGB, ColorSpaces.LAB]:
            return 3

        if cs_type == ColorSpaces.DEVICE_CMYK:
            return 4

        if cs_type == ColorSpaces.ICCBASED:
            icc_profile_stream = cast(StreamObject, self.cs_obj[1].get_object()) # type: ignore
            return int(str(icc_profile_stream.get("/N")))

        if cs_type == ColorSpaces.DEVICE_N:
            return len(self.cs_obj[1])

        raise PyPdfError(f"Number of colorants of color space {self.cs_obj!s} could not be determined.")

    def get_alternate_colorspace_name(self) -> str|NameObject|None:
        """
        Returnes the alternace color space that must be defined for a
        Separation and DeviceN color space.
        In case of a ICCBased color space it is optionally.
        The alternate color space will eventually simplify the rendering
        and extraction of images to a common and widly used basic color space
        like Gray, RGB or CMYK
        """
        if self.cs_type in [ColorSpaces.SEPARATION, ColorSpaces.DEVICE_N]:
            alternate_cs = self.cs_obj[2].get_object() # type: ignore
            if isinstance(alternate_cs, NameObject):
                return alternate_cs
            if isinstance(alternate_cs, ArrayObject):
                return alternate_cs[0]
            raise PyPdfError(f"Invalid PDF object found for alternate color space in {self.cs_obj!s}.")

        if self.cs_type == ColorSpaces.ICCBASED:
            # The alternate CS in ICCBased CS is optional. Here we simplify and act as
            # there wouldn't be an alternate CS to simplify things ...
            # The implementation should return None in such a case
            number_of_colorants = self.get_colorant_count()
            if number_of_colorants == 1:
                return ColorSpaces.DEVICE_GRAY
            if number_of_colorants == 3:
                return ColorSpaces.DEVICE_RGB
            if number_of_colorants == 4:
                return ColorSpaces.DEVICE_CMYK

        if self.cs_type == ColorSpaces.INDEXED:
            # Here we return the alternate color space of the base color space
            # Only if the base color space is an Array object (i.e. Separation or DeviceN)
            # there will be an alternate CS
            base_colorspace = self.cs_obj[1].get_object() # type: ignore
            if isinstance(base_colorspace, ArrayObject):
                return ColorSpaceWrapper(base_colorspace).get_alternate_colorspace_name()

        return None

    def get_tint_transform_object(self) -> TintTransformWarpper|None:
        """
        The Tint Transform is required to comupte from the actual pixel colors
        the related color values in the alternate color space.
        Only available and required for Separation and DeviceN color space.
        """
        if self.cs_type in [ColorSpaces.DEVICE_N, ColorSpaces.SEPARATION]:
            return TintTransformWarpper(self.cs_obj[3].get_object()) # type: ignore

        if self.cs_type == ColorSpaces.INDEXED:
            base_colorspace = ColorSpaceWrapper(self.cs_obj[1].get_object()) # type: ignore
            if base_colorspace.get_colorspace_type in [ColorSpaces.DEVICE_N, ColorSpaces.SEPARATION]:
                return base_colorspace.get_tint_transform_object()

        return None

    def get_default_decode_array(self, inverted: bool = False) -> list[float]:
        """8.9.5.2 Decode arrays of ISO 32000-2:2017"""
        if self.cs_type not in [ColorSpaces.INDEXED, ColorSpaces.LAB, ColorSpaces.ICCBASED, ColorSpaces.PATTERN]:
            if inverted:
                return [1.0, 0.0] * self.get_colorant_count()
            return [0.0, 1.0] * self.get_colorant_count()

        raise PyPdfError(f"Default decode array not yet implements for color space type {self.cs_type!s}.")

