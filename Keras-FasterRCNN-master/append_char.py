appendText='../'
names = open("hello.txt",'r')
updated_names= open('hello_new.txt','a')
for name in names:
    updated_names.write(appendText + name.rstrip() + '\n')
updated_names.close()
