"""
Optional Task 3 For Extra Credit:
Suppose you want to determine the elimination-by-aspects choice of a customer choosing from among 
P products, each having A attributes. You are given the following data structures:
an A-by-P matrix containing the rating of each product on each attribute,
a vector of length A containing the importance of each attribute,
a vector of length A giving the cutoff for each attribute 
    (we are considering the general case where it is possible for the consumer to
     have different cutoffs for different attributes). Write a function that takes these 
     three data structures as input arguments and produces the elimination-by-aspects choice
    without using explicit loops like "for", "while", "repeat" or recursion.
    
You can however use implicit loops like in the "apply" family of functions in R or
"map" function of python. 
For extra Extra Credit: 
Write the function without using any "if" statement and also without using explicit loops. 
Your solution needs to use the same policy as ties and null sets as given in Q4.
"""
import random
import collections
import pandas as pd
from itertools import product 
from functools import reduce

ratings_range = 7
ratings = [i+1 for i in range(ratings_range)]
# say there are 7 attributes and the total number of 
# products are all permutations of those attribute.
a_by_p = [p for p in product(ratings, repeat=ratings_range)]
a_by_p = list(zip(*reversed(a_by_p)))
# print(len(a_by_p))

importance = random.sample(range(1, 100), ratings_range)
# Make sure its sums to 100
_sum = sum(importance)
attr_importance = [round((i/_sum)*100, 2) for i in importance]
# print(attr_importance)

cutoff = random.sample(range(ratings_range), ratings_range)
cutoff = list(map(int, cutoff))
# print(cutoff)

def len_mapper(length):
    mapper_fn = {
        0: lambda temp, acc: (acc is not None and acc[random.randrange(len(acc))]) or [],
        1: lambda temp, acc: temp[0]
    }
    return mapper_fn.get(length, lambda temp, acc: temp)


def compute(acc, element, cutoff):
    try:

        if len(acc) == 1:
            return acc

        print('  ')
        print(element)
        
        cut = element[2]



        temp = list(
            filter(
                lambda x: x > cut,
                (acc is not None and acc) or element[1]
            )
        )
        fn = len_mapper(len(temp))

        print(len(temp), fn(temp, acc))
        acc = fn(temp, acc)
    except:
        return []
    
    return acc


def optional_3(a_by_p, importance, cutoff):
    # print(importance)
    # print(cutoff)

    zipped_columns = list(zip(importance, a_by_p, range(ratings_range)))
    data = sorted(zipped_columns, key=lambda x: x[0], reverse=True)
    data = list(map(list, data))
    result = data[:]

    df = pd.DataFrame(list(map(lambda x: x[1], data[:])), columns = range(len(a_by_p[0])))
    # print(df)

    def map_length(length):
        mapper = {
            0: [],
            1: df.columns
        }
        return mapper[length]
    
    def map_all_remove(_popper):
        removes_all = {
            'True': random.choice(df.columns) 
        }
        return removes_all[_popper]

    def map_apply(record):
        try:
            val = map_length(len(df.columns))
            return val
        except:
            pass

        _, products, index = record
        cut = cutoff[index]
        pop_indx = list(
            map(
                lambda v: v[0],
                filter(
                lambda v: v[1] < cut,
                enumerate(products)
                )
            )
        )
        set_indx = list(set(pop_indx).intersection(set(df.columns)))
        try:
            try:
                untouch = map_all_remove(str(len(set_indx) == len(df.columns)))
                df.drop(df.columns.difference([untouch]), 1, inplace=True)
                print(set_indx)
            except:
                df.drop(set_indx, axis=1, inplace=True)
        except:
            pass
        return True

    _ = list(
        map(
            map_apply,
            result
        )
    )

    columns = df.columns.values
    map_random = {
        0: [],
        1: df.columns[0]
    }
    
    selected_product = map_random.get(len(columns), random.choice(df.columns))
    print(df.columns)
    print('Product: ', selected_product+1)
    return selected_product+1


def test_option_3():
    a_by_p = [
        [10, 2, 20],
        [1, 2, 20],
        [10, 2, 20],
        [10, 2, 20],
        [10, 2, 20],
        [10, 2, 20],
        [10, 2, 0]
    ]

    importance = [23, 11, 24, 42, 75, 8, 71]
    cutoff = [2, 6, 4, 0, 3, 1, 5]
    assert(optional_3(a_by_p, importance, cutoff) ==  1)

    importance = [23, 11, 24, 42, 75, 8, 10]
    assert(optional_3(a_by_p, importance, cutoff) ==  3)


    a_by_p = [
        [1, 1, 1]
    ] * 7
    importance = [23, 11, 24, 42, 75, 8, 10]
    cutoff = [1]*7
    assert(optional_3(a_by_p, importance, cutoff) is not None)

    print('Statistics')
    stats = collections.defaultdict(int)
    for _ in range(100):
        importance = [1]*7
        stats[optional_3(a_by_p, importance, cutoff)] += 1
    
    print(stats)


if __name__ == '__main__':
    test_option_3()
    # optional_3(a_by_p, importance, cutoff)

    