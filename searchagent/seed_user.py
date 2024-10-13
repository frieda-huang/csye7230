import os

import bcrypt
from dotenv import load_dotenv
from searchagent.db_connection import Session
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

with Session.begin() as session:
    session.add(new_user)
