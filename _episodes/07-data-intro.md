---

title: "An Introduction to the Pandas Library"
teaching: 10
exercises: 0
questions:
- "How do I maninpulate tabular data in Python?"
objectives:
- Arrive at the HackWeek with a basic overview of the Pandas library for tabular data manipulation.
keypoints:
- With Pandas, Python provides an excellent environment for the analysis of tabular data.
---

# Data - an introduction to the world of Pandas

**Note:** This is an edited version of [Cliburn Chan's](http://people.duke.edu/~ccc14/sta-663-2017/07_Data.html) original tutorial, as part of his Stat-663 course at Duke.  All changes remain licensed as the original, under the terms of the MIT license.

Additionally, sections have been merged from [Chris Fonnesbeck's Pandas tutorial from the NGCM Summer Academy](https://github.com/fonnesbeck/ngcm_pandas_2017/blob/master/notebooks/1.%20Introduction%20to%20NumPy%20and%20Pandas.ipynb), which are licensed under [CC0 terms](https://creativecommons.org/share-your-work/public-domain/cc0) (aka 'public domain').

## Resources

- [The Introduction to Pandas chapter](http://proquest.safaribooksonline.com/9781491957653/pandas_html) in the Python for Data Analysis book by Wes McKinney is essential reading for this topic.  This is the [companion notebook](https://github.com/wesm/pydata-book/blob/2nd-edition/ch05.ipynb) for that chapter.
- [Pandas documentation](http://pandas.pydata.org/pandas-docs/stable/)
- [QGrid](https://github.com/quantopian/qgrid)


## Pandas

**pandas** is a Python package providing fast, flexible, and expressive data structures designed to work with *relational* or *labeled* data both. It is a fundamental high-level building block for doing practical, real world data analysis in Python. 

pandas is well suited for:

- **Tabular** data with heterogeneously-typed columns, as you might find in an SQL table or Excel spreadsheet
- Ordered and unordered (not necessarily fixed-frequency) **time series** data.
- Arbitrary **matrix** data with row and column labels

Virtually any statistical dataset, labeled or unlabeled, can be converted to a pandas data structure for cleaning, transformation, and analysis.


### Key features
    
- Easy handling of **missing data**
- **Size mutability**: columns can be inserted and deleted from DataFrame and higher dimensional objects
- Automatic and explicit **data alignment**: objects can be explicitly aligned to a set of labels, or the data can be aligned automatically
- Powerful, flexible **group by functionality** to perform split-apply-combine operations on data sets
- Intelligent label-based **slicing, fancy indexing, and subsetting** of large data sets
- Intuitive **merging and joining** data sets
- Flexible **reshaping and pivoting** of data sets
- **Hierarchical labeling** of axes
- Robust **IO tools** for loading data from flat files, Excel files, databases, and HDF5
- **Time series functionality**: date range generation and frequency conversion, moving window statistics, moving window linear regressions, date shifting and lagging, etc.


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandas import Series, DataFrame
```


```python
plt.style.use('seaborn-dark')
```

## Working with Series

* A pandas Series is a generationalization of 1d numpy array
* A series has an *index* that labels each element in the vector.
* A `Series` can be thought of as an ordered key-value store.


```python
np.array(range(5,10))
```




    array([5, 6, 7, 8, 9])




```python
x = Series(range(5,10))
```


```python
x
```




    0    5
    1    6
    2    7
    3    8
    4    9
    dtype: int64



### We can treat Series objects much like numpy vectors


```python
x.sum(), x.mean(), x.std()
```




    (35, 7.0, 1.5811388300841898)




```python
x**2
```




    0    25
    1    36
    2    49
    3    64
    4    81
    dtype: int64




```python
x[x >= 8]
```




    3    8
    4    9
    dtype: int64



### Series can also contain more information than numpy vectors

#### You can always use standard positional indexing


```python
x[1:4]
```




    1    6
    2    7
    3    8
    dtype: int64



#### Series index

But you can also assign labeled indexes.


```python
x.index = list('abcde')
x
```




    a    5
    b    6
    c    7
    d    8
    e    9
    dtype: int64



#### Note that with labels, the end index is included


```python
x['b':'d']
```




    b    6
    c    7
    d    8
    dtype: int64



#### Even when you have a labeled index, positional arguments still work


```python
x[1:4]
```




    b    6
    c    7
    d    8
    dtype: int64



#### Working with missing data

Missing data is indicated with NaN (not a number).


```python
y = Series([10, np.nan, np.nan, 13, 14])
y
```




    0    10.0
    1     NaN
    2     NaN
    3    13.0
    4    14.0
    dtype: float64



#### Concatenating two series


```python
z = pd.concat([x, y])
z
```




    a     5.0
    b     6.0
    c     7.0
    d     8.0
    e     9.0
    0    10.0
    1     NaN
    2     NaN
    3    13.0
    4    14.0
    dtype: float64



#### Reset index to default


```python
z = z.reset_index(drop=True)
z
```




    0     5.0
    1     6.0
    2     7.0
    3     8.0
    4     9.0
    5    10.0
    6     NaN
    7     NaN
    8    13.0
    9    14.0
    dtype: float64




```python
z**2
```




    0     25.0
    1     36.0
    2     49.0
    3     64.0
    4     81.0
    5    100.0
    6      NaN
    7      NaN
    8    169.0
    9    196.0
    dtype: float64



#### `pandas` aggregate functions ignore missing data


```python
z.sum(), z.mean(), z.std()
```




    (72.0, 9.0, 3.2071349029490928)



#### Selecting missing values


```python
z[z.isnull()]
```




    6   NaN
    7   NaN
    dtype: float64



#### Selecting non-missing values


```python
z[z.notnull()]
```




    0     5.0
    1     6.0
    2     7.0
    3     8.0
    4     9.0
    5    10.0
    8    13.0
    9    14.0
    dtype: float64



#### Replacement of missing values


```python
z.fillna(0)
```




    0     5.0
    1     6.0
    2     7.0
    3     8.0
    4     9.0
    5    10.0
    6     0.0
    7     0.0
    8    13.0
    9    14.0
    dtype: float64




```python
z.fillna(method='ffill')
```




    0     5.0
    1     6.0
    2     7.0
    3     8.0
    4     9.0
    5    10.0
    6    10.0
    7    10.0
    8    13.0
    9    14.0
    dtype: float64




```python
z.fillna(method='bfill')
```




    0     5.0
    1     6.0
    2     7.0
    3     8.0
    4     9.0
    5    10.0
    6    13.0
    7    13.0
    8    13.0
    9    14.0
    dtype: float64




```python
z.fillna(z.mean())
```




    0     5.0
    1     6.0
    2     7.0
    3     8.0
    4     9.0
    5    10.0
    6     9.0
    7     9.0
    8    13.0
    9    14.0
    dtype: float64



#### Working with dates / times

We will see more date/time handling in the DataFrame section.


```python
z.index = pd.date_range('01-Jan-2016', periods=len(z))
```


```python
z
```




    2016-01-01     5.0
    2016-01-02     6.0
    2016-01-03     7.0
    2016-01-04     8.0
    2016-01-05     9.0
    2016-01-06    10.0
    2016-01-07     NaN
    2016-01-08     NaN
    2016-01-09    13.0
    2016-01-10    14.0
    Freq: D, dtype: float64



#### Intelligent aggregation over datetime ranges


```python
z.resample('W').sum()
```




    2016-01-03    18.0
    2016-01-10    54.0
    Freq: W-SUN, dtype: float64



#### Formatting datetime objects (see http://strftime.org)


```python
z.index.strftime('%b %d, %Y')
```




    array(['Jan 01, 2016', 'Jan 02, 2016', 'Jan 03, 2016', 'Jan 04, 2016',
           'Jan 05, 2016', 'Jan 06, 2016', 'Jan 07, 2016', 'Jan 08, 2016',
           'Jan 09, 2016', 'Jan 10, 2016'],
          dtype='<U12')



### DataFrames

Inevitably, we want to be able to store, view and manipulate data that is *multivariate*, where for every index there are multiple fields or columns of data, often of varying data type.

A `DataFrame` is a tabular data structure, encapsulating multiple series like columns in a spreadsheet.  It is directly inspired by the R DataFrame.

### Titanic data


```python
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
titanic = pd.read_csv(url)
titanic.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic.shape
```




    (891, 15)




```python
titanic.size
```




    13365




```python
titanic.columns
```




    Index(['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
           'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town',
           'alive', 'alone'],
          dtype='object')




```python
# For display purposes, we will drop some columns
titanic = titanic[['survived', 'sex', 'age', 'fare',
                   'embarked', 'class', 'who', 'deck', 'embark_town',]]
```


```python
titanic.dtypes
```




    survived         int64
    sex             object
    age            float64
    fare           float64
    embarked        object
    class           object
    who             object
    deck            object
    embark_town     object
    dtype: object



### Summarizing a data frame


```python
titanic.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>age</th>
      <th>fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.383838</td>
      <td>29.699118</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.486592</td>
      <td>14.526497</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>20.125000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>28.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>38.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>80.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>male</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Cherbourg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>female</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>male</td>
      <td>35.0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>male</td>
      <td>NaN</td>
      <td>8.4583</td>
      <td>Q</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Queenstown</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>male</td>
      <td>54.0</td>
      <td>51.8625</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>E</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>male</td>
      <td>2.0</td>
      <td>21.0750</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>female</td>
      <td>27.0</td>
      <td>11.1333</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>female</td>
      <td>14.0</td>
      <td>30.0708</td>
      <td>C</td>
      <td>Second</td>
      <td>child</td>
      <td>NaN</td>
      <td>Cherbourg</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>female</td>
      <td>4.0</td>
      <td>16.7000</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>G</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>female</td>
      <td>58.0</td>
      <td>26.5500</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>male</td>
      <td>20.0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>male</td>
      <td>39.0</td>
      <td>31.2750</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>female</td>
      <td>14.0</td>
      <td>7.8542</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>female</td>
      <td>55.0</td>
      <td>16.0000</td>
      <td>S</td>
      <td>Second</td>
      <td>woman</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>male</td>
      <td>2.0</td>
      <td>29.1250</td>
      <td>Q</td>
      <td>Third</td>
      <td>child</td>
      <td>NaN</td>
      <td>Queenstown</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>male</td>
      <td>NaN</td>
      <td>13.0000</td>
      <td>S</td>
      <td>Second</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>female</td>
      <td>31.0</td>
      <td>18.0000</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>female</td>
      <td>NaN</td>
      <td>7.2250</td>
      <td>C</td>
      <td>Third</td>
      <td>woman</td>
      <td>NaN</td>
      <td>Cherbourg</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic.tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>male</td>
      <td>27.0</td>
      <td>13.00</td>
      <td>S</td>
      <td>Second</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>female</td>
      <td>19.0</td>
      <td>30.00</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>B</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>female</td>
      <td>NaN</td>
      <td>23.45</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>male</td>
      <td>26.0</td>
      <td>30.00</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>C</td>
      <td>Cherbourg</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>male</td>
      <td>32.0</td>
      <td>7.75</td>
      <td>Q</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Queenstown</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic.columns
```




    Index(['survived', 'sex', 'age', 'fare', 'embarked', 'class', 'who', 'deck',
           'embark_town'],
          dtype='object')




```python
titanic.index
```




    RangeIndex(start=0, stop=891, step=1)



### Indexing

The default indexing mode for dataframes with `df[X]` is to access the DataFrame's *columns*:


```python
titanic[['sex', 'age', 'class']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sex</th>
      <th>age</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>male</td>
      <td>22.0</td>
      <td>Third</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>38.0</td>
      <td>First</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>26.0</td>
      <td>Third</td>
    </tr>
    <tr>
      <th>3</th>
      <td>female</td>
      <td>35.0</td>
      <td>First</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>35.0</td>
      <td>Third</td>
    </tr>
  </tbody>
</table>
</div>



#### Using the `iloc` helper for indexing


```python
titanic.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>male</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Cherbourg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>female</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic.iloc
```




    <pandas.core.indexing._iLocIndexer at 0x106cecb70>




```python
titanic.iloc[0]
```




    survived                 0
    sex                   male
    age                     22
    fare                  7.25
    embarked                 S
    class                Third
    who                    man
    deck                   NaN
    embark_town    Southampton
    Name: 0, dtype: object




```python
titanic.iloc[0:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>male</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Cherbourg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>female</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>male</td>
      <td>35.0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic.iloc[ [0, 10, 1, 5] ]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>male</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>female</td>
      <td>4.0</td>
      <td>16.7000</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>G</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Cherbourg</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>male</td>
      <td>NaN</td>
      <td>8.4583</td>
      <td>Q</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Queenstown</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic.iloc[10:15]['age']
```




    10     4.0
    11    58.0
    12    20.0
    13    39.0
    14    14.0
    Name: age, dtype: float64




```python
titanic.iloc[10:15][  ['age'] ]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>58.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>20.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>39.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
titanic[titanic. < 2]
```


      File "<ipython-input-47-0c88d841b460>", line 1
        titanic[titanic. < 2]
                         ^
    SyntaxError: invalid syntax




```python
titanic["new column"] = 0
```


```python
titanic["new column"][:10]
```




    0    0
    1    0
    2    0
    3    0
    4    0
    5    0
    6    0
    7    0
    8    0
    9    0
    Name: new column, dtype: int64




```python
titanic[titanic.age < 2].index
```




    Int64Index([78, 164, 172, 183, 305, 381, 386, 469, 644, 755, 788, 803, 827,
                831],
               dtype='int64')




```python
df = pd.DataFrame(dict(name=['Alice', 'Bob'], age=[20, 30]), 
                  columns = ['name', 'age'],  # enforce column order
                  index=pd.Series([123, 989], name='id'))
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>123</th>
      <td>Alice</td>
      <td>20</td>
    </tr>
    <tr>
      <th>989</th>
      <td>Bob</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>



### `.iloc` vs `.loc`

These are two accessors with a key difference:

* `.iloc` indexes *positionally*
* `.loc` indexes *by label*


```python
#df[0]  # error
#df[123] # error
df.iloc[0]
```




    name    Alice
    age        20
    Name: 123, dtype: object




```python
df.loc[123]
```




    name    Alice
    age        20
    Name: 123, dtype: object




```python
df.loc[ [123] ]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>123</th>
      <td>Alice</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



#### Sorting and ordering data


```python
titanic.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>male</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>female</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>male</td>
      <td>35.0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



The `sort_index` method is designed to sort a DataFrame by either its index or its columns:


```python
titanic.sort_index(ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>male</td>
      <td>32.0</td>
      <td>7.75</td>
      <td>Q</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>male</td>
      <td>26.0</td>
      <td>30.00</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>female</td>
      <td>NaN</td>
      <td>23.45</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>female</td>
      <td>19.0</td>
      <td>30.00</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>B</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>male</td>
      <td>27.0</td>
      <td>13.00</td>
      <td>S</td>
      <td>Second</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Since the Titanic index is already sorted, it's easier to illustrate how to use it for the index with a small test DF:


```python
df = pd.DataFrame([1, 2, 3, 4, 5], index=[100, 29, 234, 1, 150], columns=['A'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2</td>
    </tr>
    <tr>
      <th>234</th>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
    </tr>
    <tr>
      <th>150</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sort_index() # same as df.sort_index('index')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1</td>
    </tr>
    <tr>
      <th>150</th>
      <td>5</td>
    </tr>
    <tr>
      <th>234</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



Pandas also makes it easy to sort on the *values* of the DF:


```python
titanic.sort_values('age', ascending=True).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>803</th>
      <td>1</td>
      <td>male</td>
      <td>0.42</td>
      <td>8.5167</td>
      <td>C</td>
      <td>Third</td>
      <td>child</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>755</th>
      <td>1</td>
      <td>male</td>
      <td>0.67</td>
      <td>14.5000</td>
      <td>S</td>
      <td>Second</td>
      <td>child</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>644</th>
      <td>1</td>
      <td>female</td>
      <td>0.75</td>
      <td>19.2583</td>
      <td>C</td>
      <td>Third</td>
      <td>child</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>469</th>
      <td>1</td>
      <td>female</td>
      <td>0.75</td>
      <td>19.2583</td>
      <td>C</td>
      <td>Third</td>
      <td>child</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>1</td>
      <td>male</td>
      <td>0.83</td>
      <td>29.0000</td>
      <td>S</td>
      <td>Second</td>
      <td>child</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



And we can sort on more than one column in a single call:


```python
titanic.sort_values(['survived', 'age'], ascending=[True, True]).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>164</th>
      <td>0</td>
      <td>male</td>
      <td>1.0</td>
      <td>39.6875</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>386</th>
      <td>0</td>
      <td>male</td>
      <td>1.0</td>
      <td>46.9000</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>male</td>
      <td>2.0</td>
      <td>21.0750</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>male</td>
      <td>2.0</td>
      <td>29.1250</td>
      <td>Q</td>
      <td>Third</td>
      <td>child</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>119</th>
      <td>0</td>
      <td>female</td>
      <td>2.0</td>
      <td>31.2750</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



*Note:* both the index and the columns can be named:


```python
t = titanic.sort_values(['survived', 'age'], ascending=[True, False])
t.index.name = 'id'
t.columns.name = 'attributes'
t.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>attributes</th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>851</th>
      <td>0</td>
      <td>male</td>
      <td>74.0</td>
      <td>7.7750</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0</td>
      <td>male</td>
      <td>71.0</td>
      <td>34.6542</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>A</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>493</th>
      <td>0</td>
      <td>male</td>
      <td>71.0</td>
      <td>49.5042</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>116</th>
      <td>0</td>
      <td>male</td>
      <td>70.5</td>
      <td>7.7500</td>
      <td>Q</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>672</th>
      <td>0</td>
      <td>male</td>
      <td>70.0</td>
      <td>10.5000</td>
      <td>S</td>
      <td>Second</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Grouping data


```python
sex_class = titanic.groupby(['sex', 'class'])
```

What is a GroubBy object?


```python
sex_class
```




    <pandas.core.groupby.DataFrameGroupBy object at 0x101c3b860>




```python
from IPython.display import display

for name, group in sex_class:
    print('name:', name, '\ngroup:\n')
    display(group.head(2))
```

    name: ('female', 'First') 
    group:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>attributes</th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    name: ('female', 'Second') 
    group:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>attributes</th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>female</td>
      <td>14.0</td>
      <td>30.0708</td>
      <td>C</td>
      <td>Second</td>
      <td>child</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>female</td>
      <td>55.0</td>
      <td>16.0000</td>
      <td>S</td>
      <td>Second</td>
      <td>woman</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    name: ('female', 'Third') 
    group:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>attributes</th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>female</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>female</td>
      <td>27.0</td>
      <td>11.1333</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    name: ('male', 'First') 
    group:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>attributes</th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>male</td>
      <td>54.0</td>
      <td>51.8625</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>E</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>male</td>
      <td>28.0</td>
      <td>35.5000</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>A</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    name: ('male', 'Second') 
    group:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>attributes</th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>male</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>S</td>
      <td>Second</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>male</td>
      <td>35.0</td>
      <td>26.0</td>
      <td>S</td>
      <td>Second</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    name: ('male', 'Third') 
    group:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>attributes</th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>male</td>
      <td>22.0</td>
      <td>7.25</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>male</td>
      <td>35.0</td>
      <td>8.05</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
sex_class.get_group(('female', 'Second')).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>attributes</th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>female</td>
      <td>14.0</td>
      <td>30.0708</td>
      <td>C</td>
      <td>Second</td>
      <td>child</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>female</td>
      <td>55.0</td>
      <td>16.0000</td>
      <td>S</td>
      <td>Second</td>
      <td>woman</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0</td>
      <td>female</td>
      <td>27.0</td>
      <td>21.0000</td>
      <td>S</td>
      <td>Second</td>
      <td>woman</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1</td>
      <td>female</td>
      <td>3.0</td>
      <td>41.5792</td>
      <td>C</td>
      <td>Second</td>
      <td>child</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>1</td>
      <td>female</td>
      <td>29.0</td>
      <td>26.0000</td>
      <td>S</td>
      <td>Second</td>
      <td>woman</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



The GroubBy object has a number of aggregation methods that will then compute summary statistics over the group members, e.g.:


```python
sex_class.count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>attributes</th>
      <th>survived</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
    <tr>
      <th>sex</th>
      <th>class</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">female</th>
      <th>First</th>
      <td>94</td>
      <td>85</td>
      <td>94</td>
      <td>92</td>
      <td>94</td>
      <td>81</td>
      <td>92</td>
      <td>94</td>
    </tr>
    <tr>
      <th>Second</th>
      <td>76</td>
      <td>74</td>
      <td>76</td>
      <td>76</td>
      <td>76</td>
      <td>10</td>
      <td>76</td>
      <td>76</td>
    </tr>
    <tr>
      <th>Third</th>
      <td>144</td>
      <td>102</td>
      <td>144</td>
      <td>144</td>
      <td>144</td>
      <td>6</td>
      <td>144</td>
      <td>144</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">male</th>
      <th>First</th>
      <td>122</td>
      <td>101</td>
      <td>122</td>
      <td>122</td>
      <td>122</td>
      <td>94</td>
      <td>122</td>
      <td>122</td>
    </tr>
    <tr>
      <th>Second</th>
      <td>108</td>
      <td>99</td>
      <td>108</td>
      <td>108</td>
      <td>108</td>
      <td>6</td>
      <td>108</td>
      <td>108</td>
    </tr>
    <tr>
      <th>Third</th>
      <td>347</td>
      <td>253</td>
      <td>347</td>
      <td>347</td>
      <td>347</td>
      <td>6</td>
      <td>347</td>
      <td>347</td>
    </tr>
  </tbody>
</table>
</div>



#### Why Kate Winslett survived and Leonardo DiCaprio didn't


```python
sex_class.mean()[['survived']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>attributes</th>
      <th>survived</th>
    </tr>
    <tr>
      <th>sex</th>
      <th>class</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">female</th>
      <th>First</th>
      <td>0.968085</td>
    </tr>
    <tr>
      <th>Second</th>
      <td>0.921053</td>
    </tr>
    <tr>
      <th>Third</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">male</th>
      <th>First</th>
      <td>0.368852</td>
    </tr>
    <tr>
      <th>Second</th>
      <td>0.157407</td>
    </tr>
    <tr>
      <th>Third</th>
      <td>0.135447</td>
    </tr>
  </tbody>
</table>
</div>



#### Of the females who were in first class, count the number from each embarking town


```python
sex_class.get_group(('female', 'First')).groupby('embark_town').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>attributes</th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>new column</th>
    </tr>
    <tr>
      <th>embark_town</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cherbourg</th>
      <td>43</td>
      <td>43</td>
      <td>38</td>
      <td>43</td>
      <td>43</td>
      <td>43</td>
      <td>43</td>
      <td>35</td>
      <td>43</td>
    </tr>
    <tr>
      <th>Queenstown</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Southampton</th>
      <td>48</td>
      <td>48</td>
      <td>44</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>43</td>
      <td>48</td>
    </tr>
  </tbody>
</table>
</div>



Since `count` counts non-missing data, we're really interested in the maximum value for each row, which we can obtain directly:


```python
sex_class.get_group(('female', 'First')).groupby('embark_town').count().max('columns')
```




    embark_town
    Cherbourg      43
    Queenstown      1
    Southampton    48
    dtype: int64



#### Cross-tabulation


```python
pd.crosstab(titanic.survived, titanic['class'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>class</th>
      <th>First</th>
      <th>Second</th>
      <th>Third</th>
    </tr>
    <tr>
      <th>survived</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>80</td>
      <td>97</td>
      <td>372</td>
    </tr>
    <tr>
      <th>1</th>
      <td>136</td>
      <td>87</td>
      <td>119</td>
    </tr>
  </tbody>
</table>
</div>



#### We can also get multiple summaries at the same time

The `agg` method is the most flexible, as it allows us to specify directly which functions we want to call, and where:


```python
def my_func(x):
    return np.max(x)
```


```python
mapped_funcs = {'embarked': 'count', 
                'age': ('mean', 'median', my_func), 
                'survived': sum}

sex_class.get_group(('female', 'First')).groupby('embark_town').agg(mapped_funcs)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>embarked</th>
      <th colspan="3" halign="left">age</th>
      <th>survived</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>my_func</th>
      <th>sum</th>
    </tr>
    <tr>
      <th>embark_town</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cherbourg</th>
      <td>43</td>
      <td>36.052632</td>
      <td>37.0</td>
      <td>60.0</td>
      <td>42</td>
    </tr>
    <tr>
      <th>Queenstown</th>
      <td>1</td>
      <td>33.000000</td>
      <td>33.0</td>
      <td>33.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Southampton</th>
      <td>48</td>
      <td>32.704545</td>
      <td>33.0</td>
      <td>63.0</td>
      <td>46</td>
    </tr>
  </tbody>
</table>
</div>



### Making plots with `pandas`

Note: you may need to run

```
pip install pandas-datareader
```

to install the specialized readers.


```python
from pandas_datareader import data as web
import datetime
```


```python
try:
    apple = pd.read_csv('data/apple.csv', index_col=0, parse_dates=True)
except:
    apple = web.DataReader('AAPL', 'yahoo', 
                            start = datetime.datetime(2015, 1, 1),
                            end = datetime.datetime(2015, 12, 31))
    # Let's save this data to a CSV file so we don't need to re-download it on every run:
    apple.to_csv('data/apple.csv')
```


```python
apple.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-02</th>
      <td>111.389999</td>
      <td>111.440002</td>
      <td>107.349998</td>
      <td>109.330002</td>
      <td>103.866470</td>
      <td>53204600</td>
    </tr>
    <tr>
      <th>2015-01-05</th>
      <td>108.290001</td>
      <td>108.650002</td>
      <td>105.410004</td>
      <td>106.250000</td>
      <td>100.940392</td>
      <td>64285500</td>
    </tr>
    <tr>
      <th>2015-01-06</th>
      <td>106.540001</td>
      <td>107.430000</td>
      <td>104.629997</td>
      <td>106.260002</td>
      <td>100.949890</td>
      <td>65797100</td>
    </tr>
    <tr>
      <th>2015-01-07</th>
      <td>107.199997</td>
      <td>108.199997</td>
      <td>106.699997</td>
      <td>107.750000</td>
      <td>102.365440</td>
      <td>40105900</td>
    </tr>
    <tr>
      <th>2015-01-08</th>
      <td>109.230003</td>
      <td>112.150002</td>
      <td>108.699997</td>
      <td>111.889999</td>
      <td>106.298531</td>
      <td>59364500</td>
    </tr>
  </tbody>
</table>
</div>




```python
apple.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-12-24</th>
      <td>109.000000</td>
      <td>109.000000</td>
      <td>107.949997</td>
      <td>108.029999</td>
      <td>104.380112</td>
      <td>13570400</td>
    </tr>
    <tr>
      <th>2015-12-28</th>
      <td>107.589996</td>
      <td>107.690002</td>
      <td>106.180000</td>
      <td>106.820000</td>
      <td>103.210999</td>
      <td>26704200</td>
    </tr>
    <tr>
      <th>2015-12-29</th>
      <td>106.959999</td>
      <td>109.430000</td>
      <td>106.860001</td>
      <td>108.739998</td>
      <td>105.066116</td>
      <td>30931200</td>
    </tr>
    <tr>
      <th>2015-12-30</th>
      <td>108.580002</td>
      <td>108.699997</td>
      <td>107.180000</td>
      <td>107.320000</td>
      <td>103.694107</td>
      <td>25213800</td>
    </tr>
    <tr>
      <th>2015-12-31</th>
      <td>107.010002</td>
      <td>107.029999</td>
      <td>104.820000</td>
      <td>105.260002</td>
      <td>101.703697</td>
      <td>40635300</td>
    </tr>
  </tbody>
</table>
</div>



Let's save this data to a CSV file so we don't need to re-download it on every run:


```python
f, ax = plt.subplots()
apple.plot.line(y='Close', marker='o', markersize=3, linewidth=0.5, ax=ax);
```


![png](07-data-intro_files/07-data-intro_119_0.png)



```python
f.suptitle("Apple stock in 2015")
f
```




![png](07-data-intro_files/07-data-intro_120_0.png)




```python
# Zoom in on large drop in August
aug = apple['2015-08-01':'2015-08-30']
aug.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-08-03</th>
      <td>121.500000</td>
      <td>122.570000</td>
      <td>117.519997</td>
      <td>118.440002</td>
      <td>113.437157</td>
      <td>69976000</td>
    </tr>
    <tr>
      <th>2015-08-04</th>
      <td>117.419998</td>
      <td>117.699997</td>
      <td>113.250000</td>
      <td>114.639999</td>
      <td>109.797668</td>
      <td>124138600</td>
    </tr>
    <tr>
      <th>2015-08-05</th>
      <td>112.949997</td>
      <td>117.440002</td>
      <td>112.099998</td>
      <td>115.400002</td>
      <td>110.525574</td>
      <td>99312600</td>
    </tr>
    <tr>
      <th>2015-08-06</th>
      <td>115.970001</td>
      <td>116.500000</td>
      <td>114.120003</td>
      <td>115.129997</td>
      <td>110.766090</td>
      <td>52903000</td>
    </tr>
    <tr>
      <th>2015-08-07</th>
      <td>114.580002</td>
      <td>116.250000</td>
      <td>114.500000</td>
      <td>115.519997</td>
      <td>111.141319</td>
      <td>38670400</td>
    </tr>
  </tbody>
</table>
</div>




```python
aug.plot.line(y=['High', 'Low', 'Open', 'Close'], marker='o', markersize=10, linewidth=1);
```

    /Users/fperez/usr/conda/envs/s159/lib/python3.6/site-packages/pandas/plotting/_core.py:1714: UserWarning: Pandas doesn't allow columns to be created via a new attribute name - see https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access
      series.name = label



![png](07-data-intro_files/07-data-intro_122_1.png)


## Data conversions

One of the nicest features of `pandas` is the ease of converting tabular data across different storage formats. We will illustrate by converting the `titanic` dataframe into multiple formats.

### CSV


```python
titanic.to_csv('titanic.csv', index=False)
```


```python
t1 = pd.read_csv('titanic.csv')
t1.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>male</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Excel

You may need to first install openpyxl:

```
pip install openpyxl
```


```python
t1.to_excel('titanic.xlsx')
```


```python
t2 = pd.read_excel('titanic.xlsx')
t2.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>male</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Relational Database


```python
import sqlite3

con = sqlite3.connect('titanic.db')
t2.to_sql('titanic', con, index=False, if_exists='replace')
```

    /Users/fperez/usr/conda/envs/s159/lib/python3.6/site-packages/pandas/core/generic.py:1534: UserWarning: The spaces in these column names will not be changed. In pandas versions < 0.14, spaces were converted to underscores.
      chunksize=chunksize, dtype=dtype)



```python
t3 = pd.read_sql('select * from titanic', con)
t3.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>male</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>None</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### JSON


```python
t3.to_json('titanic.json')
```


```python
t4 = pd.read_json('titanic.json')
t4.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>class</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>embarked</th>
      <th>fare</th>
      <th>new column</th>
      <th>sex</th>
      <th>survived</th>
      <th>who</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>Third</td>
      <td>None</td>
      <td>Southampton</td>
      <td>S</td>
      <td>7.2500</td>
      <td>0</td>
      <td>male</td>
      <td>0</td>
      <td>man</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>First</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>C</td>
      <td>71.2833</td>
      <td>0</td>
      <td>female</td>
      <td>1</td>
      <td>woman</td>
    </tr>
  </tbody>
</table>
</div>




```python
t4 = t4[t3.columns]
t4.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>male</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>None</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### HDF5

The [HDF5 format](http://proquest.safaribooksonline.com/book/physics/9781491901564/10dot-storing-data-files-and-hdf5/chp_storing_data_html) was designed in the Earth Sciences community but it can be an excellent general purpose tool. It's efficient and type-safe, so you can store complex dataframes in it and recover them back without information loss, using the `to_hdf` method:


```python
t4.to_hdf('titanic.h5', 'titanic')
```

    /Users/fperez/usr/conda/envs/s159/lib/python3.6/site-packages/pandas/core/generic.py:1471: PerformanceWarning: 
    your performance may suffer as PyTables will pickle object types that it cannot
    map directly to c-types [inferred_type->mixed,key->block1_values] [items->['sex', 'embarked', 'class', 'who', 'deck', 'embark_town']]
    
      return pytables.to_hdf(path_or_buf, key, self, **kwargs)



```python
t5 = pd.read_hdf('titanic.h5', 'titanic')
t5.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>male</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>None</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Feather

You may need to install the [Feather](https://blog.cloudera.com/blog/2016/03/feather-a-fast-on-disk-format-for-data-frames-for-r-and-python-powered-by-apache-arrow) support first:

```
conda install -c conda-forge feather-format
```


```python
t6 = t5.reset_index(drop=True)
t6.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>male</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>None</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>female</td>
      <td>4.0</td>
      <td>16.7000</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>G</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>female</td>
      <td>28.0</td>
      <td>7.8958</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>None</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>male</td>
      <td>NaN</td>
      <td>7.8958</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>None</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
t6.to_feather('titanic.feather')
```


```python
t7 = pd.read_feather('titanic.feather')
t7.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>sex</th>
      <th>age</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>new column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>male</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>None</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>female</td>
      <td>4.0</td>
      <td>16.7000</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>G</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>female</td>
      <td>28.0</td>
      <td>7.8958</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>None</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>male</td>
      <td>NaN</td>
      <td>7.8958</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>None</td>
      <td>Southampton</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


