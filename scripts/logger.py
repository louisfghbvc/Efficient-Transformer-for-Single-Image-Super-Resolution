
LOG_INDENT = "          "
def msgNewlineIndent(msg):
    return msg.replace('\n','\n'+LOG_INDENT)
def title(msg, ch="="):
    print(f"{ch*10} {msg} {ch*10}")
def warn(msg):
    print(f"<warning> {msgNewlineIndent(msg)}")
def info(msg):
    print(f"[info]    {msgNewlineIndent(msg)}")
def error(msg):
    print(f"<<Error>> {msgNewlineIndent(msg)}")
    exit(-1)
