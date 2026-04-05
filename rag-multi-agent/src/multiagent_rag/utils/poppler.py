import os
import platform


def get_poppler_path():
    system = platform.system().lower()

    if system == "windows":
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        poppler_path = os.path.join(base_dir, "tools", "poppler", "windows")

        required = ["pdftoppm.exe", "pdfinfo.exe"]
        for exe in required:
            if not os.path.exists(os.path.join(poppler_path, exe)):
                raise RuntimeError(f"Missing Poppler binary: {exe}")

        return poppler_path

    elif system in ["darwin", "linux"]:
        poppler_path = "/usr/local/bin"
        if not os.path.exists(os.path.join(poppler_path, "pdftoppm")):
            raise RuntimeError("Missing Poppler binary: pdftoppm")

        return poppler_path

    else:
        raise RuntimeError(f"Unsupported OS: {system}")


if __name__ == '__main__':
    print(get_poppler_path())
