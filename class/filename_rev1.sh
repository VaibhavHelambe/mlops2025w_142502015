#!/bin/bash 

echo "First Number:"
read a 
echo "Second Number:"
read b 

echo "sum of numbers is $((a + b))"

fact=1

while [ $a -gt 1 ]
do 
	fact=$((fact * a))
	a=$((a-1))
done 

echo "Factorial of First number is:"
echo $fact
