import time
import argparse
import sys
import threading
import os
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer


def setup_console():
    """Setup console for proper Unicode handling on Windows."""
    if sys.platform == 'win32':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception as e:
            print(f"Warning: Failed to setup Windows console: {e}")


def read_tests(fname_inp: str, fname_out: str) -> Dict[str, List[int]]:
    """Read test cases from input and output files."""
    tests = {}

    try:
        if not os.path.isfile(fname_inp):
            print(f"{__name__} : error: could not open file '{fname_inp}'")
            return {}

        if not os.path.isfile(fname_out):
            print(f"{__name__} : error: could not open file '{fname_out}'")
            return {}

        with open(fname_inp, 'r', encoding='utf-8') as f:
            raw_input = f.read()

        with open(fname_out, 'r', encoding='utf-8') as f:
            outputs = [line.strip() for line in f]

        separator = "\n__ggml_vocab_test__\n"
        inputs = raw_input.split(separator)

        if len(inputs) != len(outputs):
            print(f"{__name__} : error: input and output files have different number of tests")
            return {}

        for inp, out in zip(inputs, outputs):
            tokens = [int(tok) for tok in out.split()]
            tests[inp.strip()] = tokens

        return tests

    except Exception as e:
        print(f"{__name__} : error: {e}")
        return {}


def run_tests(tokenizer, tests: Dict[str, List[int]], thread_id: int) -> bool:
    """Run tokenization tests and verify results."""
    success = True

    for test_input, expected_tokens in tests.items():
        should_print = (thread_id == 0)

        try:
            result_tokens = tokenizer.encode(test_input, add_special_tokens=False)

            if should_print:
                print(f"\nsrc: '{test_input}'")
                print(f"res: '{tokenizer.decode(result_tokens)}'")
                print(f"tok:", " ".join(str(t) for t in result_tokens))

            correct = (len(result_tokens) == len(expected_tokens))
            if correct:
                for res, exp in zip(result_tokens, expected_tokens):
                    if res != exp:
                        correct = False
                        break

            if not correct and should_print:
                print(f"{__name__} : failed test:    '{test_input}'")
                print(
                    f"{__name__} : detokenized to: '{tokenizer.decode(result_tokens)}' instead of \
                        '{tokenizer.decode(expected_tokens)}'")
                print(f"{__name__} : expected tokens: ", end='')
                for t in expected_tokens:
                    print(f"{t:6d} '{tokenizer.decode([t])}', ", end='')
                print()
                print(f"{__name__} : got tokens:      ", end='')
                for t in result_tokens:
                    print(f"{t:6d} '{tokenizer.decode([t])}', ", end='')
                print()
                success = False

        except Exception as e:
            print(f"{__name__} : error processing test '{test_input}': {e}")
            success = False

    return success


def process_text_file(tokenizer, fname: str) -> Tuple[List[int], float]:
    """Process a single text file and return tokens and processing time."""
    if not os.path.isfile(fname):
        print(f"{__name__} : error: could not open file '{fname}'")
        return [], 0.0

    try:
        print(f"{__name__} : tokenizing: '{fname}'")
        with open(fname, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            text = ''.join(lines)

        print(f"{__name__} : text size: {len(text)}")

        start_time = time.time()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        processing_time = (time.time() - start_time) * 1000

        return tokens, processing_time

    except Exception as e:
        print(f"{__name__} : error: {e}")
        return [], 0.0


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} vocab-file [text-file]")
        return 1

    vocab_file = sys.argv[1]
    text_file = sys.argv[2] if len(sys.argv) > 2 else None

    setup_console()

    print(f"{__name__} : reading vocab from: '{vocab_file}'")

    try:
        tokenizer = AutoTokenizer.from_pretrained(vocab_file)

        if text_file:
            # Process single file mode
            tokens, processing_time = process_text_file(tokenizer, text_file)
            if not tokens:
                return 1

            fname_out = text_file + '.tokpy'
            with open(fname_out, 'w', encoding='utf-8') as f:
                for token in tokens:
                    f.write(f"{token}\n")

            print(f"{__name__} : tokenized in {processing_time:.3f} ms (py)")
            print(f"{__name__} : tokens: {len(tokens)}")
            print(f"{__name__} : tokens written to '{fname_out}'")
            return 0

        else:
            # Test suite mode
            fname_inp = vocab_file + '.inp'
            fname_out = vocab_file + '.out'

            tests = read_tests(fname_inp, fname_out)
            if not tests:
                print(f"{__name__} : error: no tests found")
                return 1

            thread_count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
            threads = []
            results = []

            for i in range(thread_count):
                thread = threading.Thread(
                    target=lambda i=i: results.append(run_tests(tokenizer, tests, i)))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            success = all(results)
            print("\nTests", "passed" if success else "failed")
            return 0 if success else 3

    except Exception as e:
        print(f"{__name__} : error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
