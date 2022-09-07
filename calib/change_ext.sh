cd left_cal
for file in *.png; do 
    mv -- "$file" "${file%.png}.jpeg"
done