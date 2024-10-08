from yaspin import yaspin


def loader(text: str):
    """
    Custom spinner for terminal.

    Args:
        text (str): Text to display
    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            with yaspin(text=text, color="cyan") as sp:
                result = func(self, *args, **kwargs)
                sp.ok("✔")
                return f"\n{result}"

        return wrapper

    return decorator
