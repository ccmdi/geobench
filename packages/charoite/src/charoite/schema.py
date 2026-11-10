import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import IntEnum
from importlib.resources import files
from typing import Any, Self, get_args, get_origin

SCALARS: dict[str, type] = {
    "double": float,
    "float": float,
    "int32": int,
    "int64": int,
    "uint64": int,
    "bool": bool,
    "string": str,
}


@dataclass(frozen=True)
class Declared:
    number: int
    type: str
    repeated: bool


def declarations() -> tuple[dict[str, dict[str, Declared]], dict[str, dict[str, int]]]:
    protos = (path.read_text() for path in files("charoite").joinpath("proto").iterdir() if path.name.endswith(".proto"))
    text = re.sub(r"//.*", "", "\n".join(protos))
    messages: dict[str, dict[str, Declared]] = {}
    enums: dict[str, dict[str, int]] = {}
    for block, name, body in re.findall(r"(message|enum)\s+(\w+)\s*\{([^}]*)\}", text):
        match block:
            case "message":
                fields = re.findall(r"(optional|repeated)\s+(\w+)\s+(\w+)\s*=\s*(\d+)\s*;", body)
                messages[name] = {field: Declared(int(n), kind, label == "repeated") for label, kind, field, n in fields}
            case "enum":
                enums[name] = {value: int(n) for value, n in re.findall(r"(\w+)\s*=\s*(\d+)\s*;", body)}
    return messages, enums


MESSAGES, ENUMS = declarations()


class PanoType(IntEnum):
    OFFICIAL = ENUMS["PanoType"]["OFFICIAL"]
    UNKNOWN = ENUMS["PanoType"]["UNKNOWN"]
    USER_UPLOADED = ENUMS["PanoType"]["USER_UPLOADED"]


@dataclass(frozen=True)
class Field:
    index: int
    read: Callable[[Any], Any]

    @classmethod
    def of(cls, declared: Declared, hint: Any) -> Self:
        repeated = get_origin(hint) is tuple
        one = get_args(hint)[0] if repeated else hint
        if repeated != declared.repeated or SCALARS.get(declared.type, declared.type) not in (one, one.__name__):
            raise TypeError(f"{hint} cannot read {declared}")
        read = one if issubclass(one, Message) else lambda value: one() if value is None else one(value)
        return cls(declared.number - 1, (lambda value: tuple(map(read, value or ()))) if repeated else read)

    def __get__(self, message: "Message", owner: type) -> Any:
        return self.read(message.wire[self.index] if self.index < len(message.wire) else None)


class Message:
    __slots__ = ("wire",)

    def __init__(self, wire: list[Any] | None) -> None:
        self.wire = wire or []

    def __init_subclass__(cls) -> None:
        declared = MESSAGES[cls.__name__]
        for name, hint in cls.__annotations__.items():
            setattr(cls, name, Field.of(declared[re.sub(r"_([a-z])", lambda m: m[1].upper(), name)], hint))

    @classmethod
    def proto(cls, data: bytes) -> Self:
        numbers = {declared.number: declared for declared in MESSAGES[cls.__name__].values()}
        wire: list[Any] = [None] * max(numbers, default=0)
        for number, value in records(data):
            match numbers.get(number), value:
                case None, _:
                    pass
                case Declared(type="string", repeated=False), bytes():
                    wire[number - 1] = value.decode()
                case Declared(type=scalar, repeated=False), int() if scalar in SCALARS:
                    wire[number - 1] = value
                case declared, _:
                    raise ValueError(f"{cls.__name__} reads no {declared} from binary")
        return cls(wire)


def varint(data: bytes, at: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[at]
        value |= (byte & 0x7F) << shift
        shift += 7
        at += 1
        if byte < 0x80:
            return value, at


def records(data: bytes) -> Iterator[tuple[int, int | bytes]]:
    at = 0
    while at < len(data):
        key, at = varint(data, at)
        match key & 7:
            case 0:
                value, at = varint(data, at)
                yield key >> 3, value
            case 2:
                length, at = varint(data, at)
                if at + length > len(data):
                    raise ValueError("truncated protobuf field")
                yield key >> 3, data[at : at + length]
                at += length
            case wire:
                raise ValueError(f"unsupported protobuf wire type {wire}")


class ResponseStatus(Message):
    code: int
    error_message: str


class ImageStatus(Message):
    code: int


class ImageKey(Message):
    frontend: int
    id: str


class ImageSize(Message):
    height: int
    width: int


class PossibleSize(Message):
    size: ImageSize


class ImageTileSize(Message):
    possible: tuple[PossibleSize, ...]


class ImageTiles(Message):
    tile_size: ImageTileSize


class LatLng(Message):
    lat: float
    lng: float


class Pov(Message):
    heading: float


class PanoLocation(Message):
    location: LatLng
    pov: Pov


class ImageInformation(Message):
    location: PanoLocation


class PanoDate(Message):
    year: int
    month: int


class ImageDate(Message):
    date: PanoDate


class ImageMetadata(Message):
    status: ImageStatus
    pano: ImageKey
    tiles: ImageTiles
    information: tuple[ImageInformation, ...]
    date: ImageDate


class SingleImageSearchResponse(Message):
    status: ResponseStatus
    result: ImageMetadata

    @property
    def image(self) -> ImageMetadata:
        return self.result


class GetMetadataResponse(Message):
    status: ResponseStatus
    metadata: tuple[ImageMetadata, ...]

    @property
    def image(self) -> ImageMetadata:
        return next(iter(self.metadata), ImageMetadata(None))
