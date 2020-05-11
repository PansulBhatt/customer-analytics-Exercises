import pandas as pd
from solution_package.mandatory import question_1, question_2, question_3
from solution_package.optional import optional_part_1

df = pd.DataFrame(question_1())
print("Question 1\n", df.head())

print("Question 2")
question_2()
question_3()
optional_part_1()
