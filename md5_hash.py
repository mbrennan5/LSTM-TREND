import hashlib
import sys


def md5_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(f"{arg}: {md5_hash(arg)}")
    else:
        for line in sys.stdin:
            text = line.rstrip("\n")
            print(f"{text}: {md5_hash(text)}")
