import re
from typing import Optional
import haiku as hk


_CAMEL_TO_SNAKE_R = re.compile(r"((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))")
camel_to_snake = lambda value: _CAMEL_TO_SNAKE_R.sub(r"_\1", value).lower()


class Module:
    def __init__(self, name: Optional[str] = None):
        if name is None:
            if hasattr(self, "name") and self.name is not None:
                # Attribute assigned by @dataclass constructor.
                name = self.name
            else:
                name = camel_to_snake(type(self).__name__)
        self.name = name

    def __call__(self, *args, **kwds):
        raise NotImplementedError

    def init_modules(self):
        pass

    def get_model(self) -> hk.Module:

        class CModel(hk.Module):
            def __init__(s):
                super().__init__(name=self.name)
                self.__class__.init_modules(self)

            def __call__(cls, *args, **kwargs):
                return self.__call__(*args, **kwargs)

        return CModel()
