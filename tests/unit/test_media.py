from unittest.mock import AsyncMock, MagicMock
from bot import media


async def test_photo_to_bytes_downloads_largest():
    photo_sizes = [MagicMock(file_size=100), MagicMock(file_size=900)]
    largest = photo_sizes[1]
    file_obj = MagicMock()
    file_obj.download_as_bytearray = AsyncMock(return_value=bytearray(b"imgbytes"))
    largest.get_file = AsyncMock(return_value=file_obj)
    out = await media.photo_to_bytes(photo_sizes)
    assert bytes(out) == b"imgbytes"
    largest.get_file.assert_awaited_once()
