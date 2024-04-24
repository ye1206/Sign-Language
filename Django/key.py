import random
import string


def generate_secret_key():
    chars = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(chars) for _ in range(50))


SECRET_KEY = generate_secret_key()
print(SECRET_KEY)
