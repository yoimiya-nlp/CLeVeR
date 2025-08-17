import re
import os
from tqdm import tqdm
import json
import glob

def search_c_files(dir_path):
    result = []
    file_list = os.listdir(dir_path)
    for file_name in tqdm(file_list):
        complete_file_name = os.path.join(dir_path, file_name)
        if os.path.isdir(complete_file_name):
            result.extend(search_c_files(complete_file_name))
        if os.path.isfile(complete_file_name):
            _, ext = os.path.splitext(complete_file_name)
            if ext in ['.c', '.cpp']:
                result.append(complete_file_name)
    return result


def extract_sources_sinks(file_content):
    pattern = r"(?:BadSource:\s*(.+?)\s*)?(?:GoodSource:\s*(.+?)\s*)?Sinks:\s*(.+?)\s*GoodSink:\s*(.+?)\s*BadSink\s*:\s*(.+?)(?=\s*Flow)"
    match = re.search(pattern, file_content, re.DOTALL)
    if match:
        bad_source, good_source, sinks, good_sink, bad_sink = match.groups()
        sources_sinks = {
            "BadSource": bad_source.strip() if bad_source else "",
            "GoodSource": good_source.strip() if good_source else "",
            "Sinks": sinks.strip() if sinks else "",
            "GoodSink": good_sink.strip() if good_sink else "",
            "BadSink": bad_sink.strip() if bad_sink else ""
        }
        return sources_sinks
    else:
        source_sinks = {
            "BadSource": "",
            "GoodSource": "",
            "Sinks": "",
            "GoodSink": "",
            "BadSink": ""
        }
        return source_sinks


def extract_functions(file_content):
    functions = {}

    pattern = r"(?P<function_name>\w+)\s*\([^)]*\)\s*\{(?P<function_body>.*?)\n\}"

    matches = re.finditer(pattern, file_content, re.DOTALL)
    for match in matches:
        function_name = match.group('function_name')
        #function_body = match.group('function_body')
        if function_name != "main":
            functions[function_name] = f"{match.group()}"

    return functions


def extract_cwe_number(function_name):
    pattern = r"CWE(\d+)_"
    match = re.search(pattern, function_name)
    if match:
        return match.group(1)
    else:
        return None
'''
cwe2desc = {"None",
            "15", "23", "36", "74", "78", "79", "89", "99", "114", "118",
            "119", "120", "121", "122", "123", "124", "126", "127", "129", "131",
            "134", "135", "170", "176", "187", "188", "190", "191", "193", "194",
            "195", "196", "197", "200", "204", "222", "223", "226", "242", "243",
            "244", "247", "248", "251", "252", "253", "256", "259", "272", "273",
            "284", "319", "321", "325", "327", "328", "329", "338", "162", "364",
            "365", "366", "367", "369", "374", "377", "378", "390", "391", "392",
            "396", "397", "398", "400", "401", "404", "411", "412", "415", "416",
            "426", "427", "440", "457", "459", "463", "464", "466", "467", "468",
            "469", "475", "476", "478", "479", "480", "481", "482", "483", "484",
            "489", "495", "500", "511", "532", "534", "535", "546", "547", "560", 
            "561", "562", "563", "570", "571", "587", "588", "590", "591", "605", 
            "606", "617", "620", "665", "666", "672", "674", "675", "680", "685", 
            "688", "690", "704", "758", "761", "762", "772", "778", "785", "787", 
            "789", "822", "824", "835", "1071", "1164"}
'''

cwe2desc = {"None": "None",
            "15":"External Control of System or Configuration Setting", "23":"Relative Path Traversal",
            "36":"Absolute Path Traversal", "74":"Injection",
            "78":"OS Command Injection", "79":"Cross-site Scripting",
            "89":"SQL Injection", "99":"Resource Injection",
            "114":"Process Control", "118":"Incorrect Access of Indexable Resource",
            "119":"Buffer Overflow", "120":"Buffer Overflow",
            "121":"Stack-based Buffer Overflow", "122":"Heap-based Buffer Overflow",
            "123":"Write-what-where Condition", "124":"Buffer Underwrite",
            "126":"Buffer Over-read", "127":"Buffer Under-read",
            "129":"Improper Validation of Array Index", "131":"Incorrect Calculation of Buffer Size",
            "134":"Use of Externally-Controlled Format String", "135":"Incorrect Calculation of Multi-Byte String Length",
            "170":"Improper Null Termination", "176":"Improper Handling of Unicode Encoding",
            "187":"Partial String Comparison", "188":"Reliance on Data/Memory Layout",
            "190":"Integer Overflow", "191":"Integer Underflow",
            "193":"Off-by-one Error", "194":"Unexpected Sign Extension",
            "195":"Signed to Unsigned Conversion Error", "196":"Unsigned to Signed Conversion Error",
            "197":"Numeric Truncation Error", "200":"Exposure of Sensitive Information",
            "204":"Observable Response Discrepancy", "222":"Truncation of Security-relevant Information",
            "223":"Omission of Security-relevant Information", "226":"Sensitive Information in Resource Not Removed Before Reuse",
            "242":"Use of Inherently Dangerous Function", "243":"Creation of chroot Jail Without Changing Working Directory",
            "244":"Heap Inspection", "247":"Reliance on DNS Lookups in a Security Decision",
            "248":"Uncaught Exception", "251":"String Management",
            "252":"Unchecked Return Value", "253":"Incorrect Check of Function Return Value",
            "256":"Plaintext Storage of a Password", "259":"Use of Hard-coded Password",
            "272":"Least Privilege Violation", "273":"Improper Check for Dropped Privileges",
            "284":"Improper Access Control", "319":"Cleartext Transmission of Sensitive Information",
            "321":"Use of Hard-coded Cryptographic Key", "325":"Missing Cryptographic Step",
            "327":"Use of a Broken or Risky Cryptographic Algorithm", "328":"Use of Weak Hash",
            "329":"Generation of Predictable IV with CBC Mode", "338":"Use of Cryptographically Weak Pseudo-Random Number Generator",
            "362":"Race Condition", "364":"Signal Handler Race Condition",
            "365":"Race Condition in Switch", "366":"Race Condition within a Thread",
            "367":"Time-of-check Time-of-use Race Condition", "369":"Divide By Zero",
            "374":"Passing Mutable Objects to an Untrusted Method", "377":"Insecure Temporary File",
            "378":"Creation of Temporary File With Insecure Permissions", "390":"Detection of Error Condition Without Action",
            "391":"Unchecked Error Condition", "392":"Missing Report of Error Condition",
            "396":"Declaration of Catch for Generic Exception", "397":"Declaration of Throws for Generic Exception",
            "398":"Code Quality", "400":"Uncontrolled Resource Consumption",
            "401":"Missing Release of Memory after Effective Lifetime", "404":"Improper Resource Shutdown or Release",
            "411":"Resource Locking Problems", "412":"Unrestricted Externally Accessible Lock",
            "415":"Double Free", "416":"Use After Free",
            "426":"Untrusted Search Path", "427":"Uncontrolled Search Path Element",
            "440":"Expected Behavior Violation", "457":"Use of Uninitialized Variable",
            "459":"Incomplete Cleanup", "463":"Deletion of Data Structure Sentinel",
            "464":"Addition of Data Structure Sentinel", "466":"Return of Pointer Value Outside of Expected Range",
            "467":"Use of sizeof() on a Pointer Type", "468":"Incorrect Pointer Scaling",
            "469":"Use of Pointer Subtraction to Determine Size", "475":"Undefined Behavior for Input to API",
            "476":"NULL Pointer Dereference", "478":"Missing Default Case in Multiple Condition Expression",
            "479":"Signal Handler Use of a Non-reentrant Function", "480":"Use of Incorrect Operator",
            "481":"Assigning instead of Comparing", "482":"Comparing instead of Assigning",
            "483":"Incorrect Block Delimitation", "484":"Omitted Break Statement in Switch",
            "489":"Active Debug Code", "495":"Private Data Structure Returned From A Public Method",
            "500":"Public Static Field Not Marked Final", "511":"Logic/Time Bomb",
            "532":"Insertion of Sensitive Information into Log File", "534":"Information Exposure Through Debug Log Files",
            "535":"Exposure of Information Through Shell Error Message",
            "546":"Suspicious Comment", "547":"Use of Hard-coded, Security-relevant Constants",
            "560":"Use of umask() with chmod-style Argument", "561":"Dead Code",
            "562":"Return of Stack Variable Address", "563":"Assignment to Variable without Use",
            "570":"Expression is Always False", "571":"Expression is Always True",
            "587":"Assignment of a Fixed Address to a Pointer", "588":"Attempt to Access Child of a Non-structure Pointer",
            "590":"Free of Memory not on the Heap", "591":"Sensitive Data Storage in Improperly Locked Memory",
            "605":"Multiple Binds to the Same Port", "606":"Unchecked Input for Loop Condition",
            "617":"Reachable Assertion", "620":"Unverified Password Change",
            "665":"Improper Initialization", "666":"Operation on Resource in Wrong Phase of Lifetime",
            "672":"Operation on a Resource after Expiration or Release", "674":"Uncontrolled Recursion",
            "675":"Multiple Operations on Resource in Single-Operation Context", "680":"Integer Overflow to Buffer Overflow",
            "685":"Function Call With Incorrect Number of Arguments", "688":"Function Call With Incorrect Variable or Reference as Argument",
            "690":"Unchecked Return Value to NULL Pointer Dereference", "704":"Incorrect Type Conversion or Cast",
            "758":"Reliance on Undefined, Unspecified, or Implementation-Defined Behavior", "761":"Free of Pointer not at Start of Buffer",
            "762":"Mismatched Memory Management Routines", "772":"Missing Release of Resource after Effective Lifetime",
            "778":"Insufficient Logging",
            "785":"Use of Path Manipulation Function without Maximum-sized Buffer", "787":"Out-of-bounds Write",
            "789":"Memory Allocation with Excessive Size Value", "822":"Untrusted Pointer Dereference",
            "824":"Access of Uninitialized Pointer", "835":"Infinite Loop",
            "1071":"Empty Code Block", "1164":"Irrelevant Code"}

def main():
    files = search_c_files('./data')
    with open("dataset.jsonl", "w") as jsonl_file:
        for file_name in tqdm(files):
            code_file = file_name
            txt_file = code_file[2:].split('.')[0] + '.txt'

            with open(code_file, 'r', errors='ignore') as cf:
                code_content = cf.read()
                code_functions = extract_functions(code_content)
                func_list = []
                for function_name, function_code in code_functions.items():
                    func = {}
                    if function_name.find('bad') != -1:
                        label = 1
                    else:
                        label = 0
                    func['func'] = function_code
                    func['name'] = function_name
                    func['label'] = str(label)
                    func['cwe_id'] = str(extract_cwe_number(function_name))
                    func_list.append(func)

            with open(txt_file, "r") as tf:
                txt_content = tf.read()
                sources_sinks = extract_sources_sinks(txt_content)
                good_description = {
                    "GoodSource": sources_sinks['GoodSource'],
                    "GoodSink": sources_sinks['GoodSink']
                }
                bad_description = {
                    "BadSource": sources_sinks['BadSource'],
                    "BadSink": sources_sinks['BadSink']
                }
                for func in func_list:
                    if int(func['label']) == 1:
                        func['source'] = bad_description['BadSource']
                        func['sink'] = bad_description['BadSink']
                    elif func['name'].find('G2B') != -1:
                        func['source'] = good_description['GoodSource']
                        func['sink'] = bad_description['BadSink']
                    elif func['name'].find('B2G') != -1:
                        func['source'] = bad_description['BadSource']
                        func['sink'] = good_description['GoodSink']
                    else:
                        func['source'] = good_description['GoodSource']
                        func['sink'] = good_description['GoodSink']
                for func in func_list:
                    if int(func['label']) == 1:
                        func['reason'] = "This function first " + func['source'] + " and then " + func['sink'] + ", which may cause a " + cwe2desc[func['cwe_id']] + "."
                    else:
                        func['reason'] = "This function first " + func['source'] + " and then " + func['sink'] + ", which is a secure function."
                    if func['source'] == "" and func['sink'] == "":
                        if int(func['label']) == 1:
                            func['reason'] = "This function may cause a " + cwe2desc[data['cwe_id']] + "."
                        else:
                            func['reason'] = "This function is a secure function."
                    
                for func in func_list:
                    json_func = json.dumps(func)
                    jsonl_file.write(json_func)
                    jsonl_file.write('\n')

def statistic_dataset():
    with open("dataset.jsonl", "r") as jsonl_file:
        num = 0
        vul = 0
        sec = 0
        for line in tqdm(jsonl_file):
            line = line.strip()
            js = json.loads(line)
            #print("demo: ", js)
            num += 1
            if int(js['label']) == 1:
                vul += 1
            else:
                sec += 1

        print("dataset num: ", num)
        print("vulnerability num: ", vul)
        print("security num: ", sec)


if __name__ == '__main__':
    main()
    # statistic_dataset()
