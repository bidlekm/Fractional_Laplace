all: project

project: BMPImage.o GaussianBlur.o proj.o
	nvcc -std=c++14 -g -o project BMPImage.o GaussianBlur.o proj.o

BMPImage.o: BMPImage.cpp
	g++ -std=c++14 -O3 -g -c BMPImage.cpp -o BMPImage.o

GaussianBlur.o: GaussianBlur.cpp
	g++ -std=c++14 -O3 -g -c GaussianBlur.cpp -o GaussianBlur.o

proj.o: proj.cu
	nvcc -std=c++14 -I /usr/local/include/eigen3 -g -c proj.cu -o proj.o

clean:
	rm -f BMPImage.o GaussianBlur.o proj.o project

