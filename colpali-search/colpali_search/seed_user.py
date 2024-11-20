import asyncio

import bcrypt
from colpali_search.config import settings
from colpali_search.database import async_session
from colpali_search.models import User
from colpali_search.utils import get_now

password = settings.seed_user_password
password_bytes = password.encode("utf-8")
password_hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())

user_dict = {
    "email": "colpalisearch@gmail.com",
    "password_hash": password_hashed,
    "created_at": get_now(),
}

new_user = User(**user_dict)


async def add_new_user():
    async with async_session.begin() as session:
        session.add(new_user)


asyncio.run(add_new_user())
