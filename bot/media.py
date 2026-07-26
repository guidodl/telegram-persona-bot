from io import BytesIO
from telegram import InputFile


async def photo_to_bytes(photo_sizes) -> bytes:
    largest = max(photo_sizes, key=lambda p: p.file_size)
    file_obj = await largest.get_file()
    return bytes(await file_obj.download_as_bytearray())


def to_voice(audio_bytes: bytes) -> InputFile:
    return InputFile(BytesIO(audio_bytes), filename="voice.ogg")


def to_photo(image_bytes: bytes) -> InputFile:
    return InputFile(BytesIO(image_bytes), filename="photo.jpg")
