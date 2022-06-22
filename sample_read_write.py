f = open('sample01.txt', 'w')
f.write('Hello World')
f.close

f = open('sample01.txt', 'r')
file_content = f.readlines()
print(file_content)
