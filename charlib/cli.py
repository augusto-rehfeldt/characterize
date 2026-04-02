import argparse
import re
import os

def choose_option(what, options_list, text=True):
    if text:
        list_as_items = [f"{i+1}) {item}\n" for i, item in enumerate(options_list)]
        items = "".join(list_as_items)
        print(f"\nWhich of the following {what} do you want to use?\n\n{items}")
    try:
        choice = int(input("Choice: "))
    except ValueError:
        return choose_option(what, options_list, False)
    else:
        if not 0 < choice <= len(options_list):
            return choose_option(what, options_list, False)
        return options_list[choice - 1]

def choose_value(what, min=False, max=False, text=True):
    if text:
        range_text = ""
        if min and max:
            range_text = f" between {min} and {max}"
        elif min:
            range_text = f" above {min}"
        elif max:
            range_text = f" below {max}"
        print(f"\nPlease enter a {what} integer value{range_text} (float values will be converted to integers by rounding down).\n")
    try:
        choice = int(input("Choice: "))
    except ValueError:
        return choose_value(what, min, max, False)
    else:
        if max and choice > max:
            print(f"Choose a value below {max+1}.")
            return choose_value(what, min, max, False)
        if min and choice < min:
            print(f"Choose a value above {min-1}.")
            return choose_value(what, min, max, False)
        return choice

def binary_choice(what, text=True):
    if text:
        print(f"\nDo you want to {what}?\n\n1) Yes\n2) No\n")
    try:
        choice = int(input("Choice: "))
    except ValueError:
        return binary_choice(what, False)
    else:
        if choice not in (1, 2):
            return binary_choice(what, False)
        return choice == 1

def input_files(text=True):
    if text:
        print(
            "\nPlease, input all file paths; or a directory containing them. "
            "Separate multiple paths using spaces. For paths containing spaces, "
            "use double (\"\") or single ('') quotes for every path.\n"
        )
    choice = input("Path/s: ")
    if not choice:
        return input_files()
    if choice.count('"') >= 2:
        paths = [x.strip() for x in re.findall(r'"([^"]*)"', choice)]
    elif choice.count("'") >= 2:
        paths = [x.strip() for x in re.findall(r"'([^']*)'", choice)]
    else:
        paths = [x.strip() for x in choice.split()]
    paths = [x for x in paths if os.path.exists(x)]
    return paths if paths else input_files()

def parse_arguments():
    def str_to_bool(value):
        return value.lower() in ('true', 't', 'yes', 'y')
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--i", type=str, nargs="*", help='input file paths MANY ["path1", "path2", ...]')
    parser.add_argument("-cr", "--cr", type=int, help="character resolution parameter ONE (1 to 4000)")
    parser.add_argument("-cl", "--cl", type=int, help="complexity level parameter ONE (1 to 4000)")
    parser.add_argument("-l", "--l", type=str, help="language parameter ONE [ascii, chinese, ...]")
    parser.add_argument("-d", "--d", type=str, help="divide parameter ONE (true/false)")
    parser.add_argument("-c", "--c", type=str, help="color parameter ONE (true/false)")
    parser.add_argument("-f", "--f", nargs="+", help="format parameter MANY [png, jpg, txt]")
    parser.add_argument("-o", "--o", type=str, help="optimize parameter ONE (true/false)")
    parser.add_argument("-ec", "--ec", type=str, help="empty character parameter ONE (true/false)")
    parser.add_argument("-tk", "--tk", type=lambda x: str_to_bool(x), help="tkinter parameter ONE (true/false)")
    args = parser.parse_args()

    input_files_arg = args.i
    cr = (True, (args.cr, args.cr)) if isinstance(args.cr, int) else (False, False)
    cl = args.cl
    l = args.l
    ec = (True, str_to_bool(args.ec)) if args.ec else (False, False)
    d = (True, str_to_bool(args.d)) if args.d else (False, False)
    c = (True, str_to_bool(args.c)) if args.c else (False, False)
    f = " ".join(args.f) if args.f else None
    o = (True, str_to_bool(args.o)) if args.o else (False, False)
    tk = args.tk

    return l, cl, c, input_files_arg, cr, ec, d, f, o, tk