""""配置加载器：单例 + AttrDict + 环境变量插值"""
import os
import re
from pathlib import Path
import yaml
from dotenv import load_dotenv

class AttrDict(dict):
    def __getattr__(self, key):
        try: return self[key]
        except KeyError: raise AttributeError(key)
    def __setattr__(self, key, value): self[key] = value

class ConfigLoader:
    _instance = None
    def __new__(cls, config_dir="./configs"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_dir="./configs"):
        if self._initialized: return
        self.config_dir = Path(config_dir)
        self._cache = {}
        load_dotenv()
        self._load_all()
        self._initialized = True

    def _load_all(self):
        for yaml_file in self.config_dir.glob("*.yaml"):
            name = yaml_file.stem.replace("_config", "")
            with open(yaml_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 正则替换环境变量，支持 ${VAR:-默认值} 语法
            content = re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1).split(":-")[0], m.group(1).split(":-")[1] if ":-" in m.group(1) else m.group(0)), content)
            
            data = yaml.safe_load(content)
            self._cache[name] = self._to_attr_dict(data)

    def _to_attr_dict(self, obj):
        if isinstance(obj, dict): return AttrDict({k: self._to_attr_dict(v) for k, v in obj.items()})
        elif isinstance(obj, list): return [self._to_attr_dict(i) for i in obj]
        return obj

    def get(self, name): return self._cache.get(name)
    def reload(self):
        self._cache.clear()
        self._load_all()