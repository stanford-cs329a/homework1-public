from cs329_hw1.tasks.math_utils import (
    strip_string,
    extract_answer,
    math_equal,
)
import timeout_decorator


class MATH500Verifier:
    def __init__(self):
        pass

    def verify(
        self, solution: str, ground_truth: str, normalize_prediction=True
    ) -> int:
        if normalize_prediction:
            extracted_answer = strip_string(extract_answer(solution, "math"))
        else:
            extracted_answer = solution
        extracted_gt_answer = ground_truth
        time_out_math_equal = timeout_decorator.timeout(2)(math_equal)
        try:
            return int(time_out_math_equal(extracted_answer, extracted_gt_answer))

        except timeout_decorator.TimeoutError:
            return 0
        except Exception as e:
            return 0

    def __call__(self, *args, **kwargs):
        return self.verify(*args, **kwargs)
