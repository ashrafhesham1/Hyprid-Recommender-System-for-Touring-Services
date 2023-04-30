import dotenv
import os

class Env:
    def __init__(self):
        self.dotenv_file = dotenv.find_dotenv()
        dotenv.load_dotenv(self.dotenv_file)
    
    def get(self,var):
        env_var = os.environ.get(var,None)
        return env_var

    def set(self,var,val):
        os.environ[var] = val
        dotenv.set_key(self.dotenv_file, var, os.environ[var])