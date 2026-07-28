from io import BytesIO
import httpx
from telegram import InputFile


async def photo_to_bytes(photo_sizes) -> bytes:
    largest = max(photo_sizes, key=lambda p: p.file_size)
    file_obj = await largest.get_file()
    return bytes(await file_obj.download_as_bytearray())


async def fetch_image(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.content


def to_audio(audio_bytes: bytes) -> InputFile:
    return InputFile(BytesIO(audio_bytes), filename="voice.mp3")


def to_photo(image_bytes: bytes) -> InputFile:
    return InputFile(BytesIO(image_bytes), filename="photo.jpg")
