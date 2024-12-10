all: project

project: BMPImage.o GaussianBlur.o proj.o
	g++ -std=c++14 -g -o project BMPImage.o GaussianBlur.o proj.o

BMPImage.o: BMPImage.cpp
	g++ -std=c++14 -O3 -g -c BMPImage.cpp -o BMPImage.o

GaussianBlur.o: GaussianBlur.cpp
	g++ -std=c++14 -O3 -fno-diagnostics-show-template-tree -isystem /usr/local/include/eigen3 -g -c GaussianBlur.cpp -o GaussianBlur.o

proj.o: proj.cpp
	g++ -std=c++14 -fno-diagnostics-show-template-tree -isystem /usr/local/include/eigen3 -g -c proj.cpp -o proj.o 

clean:
	rm -f BMPImage.o GaussianBlur.o proj.o project

