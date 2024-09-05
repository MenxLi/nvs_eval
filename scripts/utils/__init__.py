import subprocess, os

class BCOLORS:
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def check_call(cmds: list[str], env = None, error_log_file = None):
    this_env = os.environ.copy()
    this_env.update(env)
    print(f"RUNNING: {BCOLORS.OKGREEN}{' '.join(cmds)}{BCOLORS.ENDC} | ENV: {env}")
    # ret = subprocess.run(cmds, env=this_env, check=False, capture_output=True, text=True)
    ret = subprocess.run(cmds, env=this_env, check=True)
    if ret.stderr:
        print(f"{BCOLORS.FAIL}ERROR: {ret.stderr}{BCOLORS.ENDC}")
        if error_log_file is not None:
            with open(error_log_file, "w") as f:
                f.write(str(ret.stderr))
        raise subprocess.CalledProcessError(ret.returncode, cmd = ret.args, stderr = ret.stderr)

    # except subprocess.CalledProcessError as e:
    #     if error_log_file is not None:
    #         with open(error_log_file, "w") as f:
    #             f.write(str(e.with_traceback(None)))
    #     print(f"{BCOLORS.FAIL}ERROR: {e}{BCOLORS.ENDC}")
    #     raise e
    
    return ret