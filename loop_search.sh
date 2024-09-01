#!/bin/bash

queries=( "How to pack light for a trip" "in 2009 Moeller signed a minor league contract with the Baltimore Orioles" "which amplifiers have vacuum tube triode sound" "Shelby sends Phil Hill and Bruce McLaren to Le Mans" "Its enchanting coastline is all breathtaking panoramas, idyllic beaches and pools" "It starts with ginger-scented chicken-cilantro meatballs that are browned" "two distinct deserts—the Mojave and Sonoran—as it tumbles down from the heights into the Coachella Valley near Palm Springs" "The park embraces parts of two distinct deserts—the Mojave and Sonoran" "breaks the weak isospin symmetry of the electroweak interaction" "Is the Higgs Boson The God Particle" "How can we recover from loss of quorum" "The Carrera S coupe is powered by a 3.8-liter flat-six paired with a seven-speed PDK automatic transaxle" "What is a Confluent Hypergeometric function" "This tuna melt eats like a tuna grilled cheese and proves that the best sandwiches are all about contrast" )

for n in {1..100000}
do
  q=${queries[ $RANDOM % ${#queries[@]} ]}
  echo "$q"
  time ./search_client.sh $q
done

