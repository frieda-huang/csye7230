import asyncio
import os

import bcrypt
from dotenv import load_dotenv
from searchagent.db_connection import async_session
from searchagent.models import User
from searchagent.utils import get_now

load_dotenv()

password = os.getenv("USER_PASSWORD")
password_bytes = password.encode("utf-8")
password_hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())

user_dict = {
    "email": "searchagent@gmail.com",
    "password_hash": password_hashed,
    "created_at": get_now(),
}

new_user = User(**user_dict)


async def add_new_user():
    async with async_session.begin() as session:
        session.add(new_user)


asyncio.run(add_new_user())
