import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
arr3 = np.linspace(0, 1, 5) 

print("Original Array:", arr1)
print("Array + 5:", arr1 + 5)
print("Array squared:", arr1 ** 2)
print("Array multiplied by itself:", arr1 * arr1)

print("Mean:", np.mean(arr1))
print("Standard Deviation:", np.std(arr1))
print("Max:", np.max(arr1))
print("Min:", np.min(arr1))

print("Reshaped arr2 to (3,2):\n", arr2.reshape(3, 2))
print("Flattened arr2:", arr2.flatten())

matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
print("Matrix Multiplication:\n", np.dot(matrix1, matrix2))

data1 = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [85, 90, 88]
}
df1 = pd.DataFrame(data1)

data2 = {
    'Name': ['David', 'Eva'],
    'Age': [22, 28],
    'Score': [75, 92]
}
df2 = pd.DataFrame(data2)

df_concat = pd.concat([df1, df2], ignore_index=True)
print("Concatenated DataFrame (Vertical):")
print(df_concat)

df_age = df_concat[['Age']]
df_score = df_concat[['Score']]
df_horiz_concat = pd.concat([df_age, df_score], axis=1)
print("\nConcatenated DataFrame (Horizontal):")
print(df_horiz_concat)


extra_info = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Department': ['CS', 'Math', 'Physics', 'CS', 'Math']
}
df_info = pd.DataFrame(extra_info)
df_merged = pd.merge(df_concat, df_info, on='Name')
print("\nMerged DataFrame on 'Name':")
print(df_merged)

df_concat_indexed = df_concat.set_index('Name')
df_info_indexed = df_info.set_index('Name')

df_joined = df_concat_indexed.join(df_info_indexed)
print("\nJoined DataFrame using index:")
print(df_joined)

df_joined_outer = df_concat_indexed.join(df_info_indexed, how='outer')
print("\nOuter Join on index:")
print(df_joined_outer)

plt.figure()
plt.plot(df_merged['Name'], df_merged['Score'], marker='o')
plt.title('Scores of Students')
plt.xlabel('Name')
plt.ylabel('Score')
plt.grid(True)
plt.show()

plt.figure()
plt.bar(df_merged['Name'], df_merged['Age'], color='orange')
plt.title('Age of Students')
plt.xlabel('Name')
plt.ylabel('Age')
plt.show()

plt.figure()
plt.pie(df_merged['Score'], labels=df_merged['Name'], autopct='%1.1f%%', startangle=90)
plt.title('Score Distribution')
plt.show()

plt.figure()
plt.hist(df_merged['Age'], bins=5, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure()
plt.scatter(df_merged['Age'], df_merged['Score'], color='green')
plt.title('Score vs Age')
plt.xlabel('Age')
plt.ylabel('Score')
plt.grid(True)
plt.show()

dept_group = df_merged.groupby('Department')[['Age', 'Score']].mean()
dept_group.plot(kind='bar', stacked=True)
plt.title('Average Age and Score by Department')
plt.ylabel('Value')
plt.show()

plt.figure()
df_merged.boxplot(column='Score', by='Department')
plt.title('Score Distribution by Department')
plt.suptitle('')
plt.xlabel('Department')
plt.ylabel('Score')
plt.show()
