package main

import (
	"os"
	"fmt"
	"math"
	"math/rand"
	"image"
	"image/gif"
	"image/color"
)

const SIZE = 256

func _init(x [SIZE][SIZE]int) [SIZE][SIZE]int{
	for i := 0 ; i < len(x); i++{
		for j := 0 ; j < len(x[1]); j++{
			if ( rand.Float64() < 0.5){
				x[i][j] = 1 
			}else{
				x[i][j] = -1
			}
		}
	}
	return x
}
func initCalcE(x [SIZE][SIZE]int) [SIZE][SIZE]int{
	var E [SIZE][SIZE]int
	for i := 0 ; i < len(x); i++{
		for j := 0 ; j < len(x[1]); j++{
			E[i][j] = 0
			u, d, l, r := i-1, i+1, j-1, j+1
			//up
			if (u < 0){
				u = SIZE - 1
			}
			if (d >= SIZE){
				d = 0
			}
			if (l < 0){
				l = SIZE -1 
			}
			if (r >= SIZE){
				r = 0
			}
			//up			
			E[i][j] += - x[i][j] * x[u][j]
			//down
			E[i][j] += - x[i][j] * x[d][j]			
			// left
			E[i][j] += - x[i][j] * x[i][l]
			// right
			E[i][j] += - x[i][j] * x[i][r]
		}
	}
	return E
}

func deltaE(x [SIZE][SIZE]int, i int,j int)int {
	tmpE := 0
	u, d, l, r := i-1, i+1, j-1, j+1
	//up
	if (u < 0){
		u = SIZE - 1
	}
	if (d >= SIZE){
		d = 0
	}
	if (l < 0){
		l = SIZE -1 
	}
	if (r >= SIZE){
		r = 0
	}
	//up			
	tmpE += - x[i][j] * x[u][j]
	//down
	tmpE += - x[i][j] * x[d][j]			
	// left
	tmpE += - x[i][j] * x[i][l]
	// right
	tmpE += - x[i][j] * x[i][r]	

	return tmpE
}

func simulation(x [SIZE][SIZE]int, E [SIZE][SIZE]int) ([SIZE][SIZE]int ,[SIZE][SIZE]int){
	invT := 1/1.7

	for i := 0 ; i<SIZE ; i++{
		for j := 0 ; j<SIZE ; j++{
			x[i][j] = - x[i][j]
			tmpE := deltaE(x,i,j)
			if(tmpE < 0){
				E[i][j] = tmpE
			}else if( tmpE > 0){
				if(math.Exp( -2.0*float64(tmpE)*invT ) < rand.Float64()){
					x[i][j] = - x[i][j]
					E[i][j] = - tmpE
				}
			}
		}
	}
	return x , E
}

func main(){
	var x [SIZE][SIZE]int
	var E [SIZE][SIZE]int

	delay := 8
	out, _ := os.Create("test.gif")
	
	rand.Seed(1)
	
	fmt.Println("hello")
	x = _init(x)

	E = initCalcE(x)
	anim := gif.GIF{LoopCount: 100}
	
	
	/*
	for i := 0 ; i < len(x); i++{
		for j := 0 ; j < len(x[1]); j++{
			if (E[i][j] == 2){
				img.Set(i, j, color.RGBA{123, 0, 0, 255})
			}else if(E[i][j] == 4){
				img.Set(i, j, color.RGBA{255, 0, 0, 255})
			}else if(E[i][j] == -2){
				img.Set(i, j, color.RGBA{0, 0, 123, 255})
			}else if(E[i][j] == -4){
				img.Set(i, j, color.RGBA{0, 0, 255, 255})
			}else{
				img.Set(i, j, color.RGBA{0, 0, 0, 10})
			}
		}
	}
*/
	var palatte = []color.Color{color.White, color.Black}
	
	for i:=0 ; i<100 ;i++{
		fmt.Println(i)
		img := image.NewPaletted(image.Rect(0, 0, SIZE, SIZE),palatte)
		for i := 0 ; i < len(x); i++{
			for j := 0 ; j < len(x[1]); j++{
				if (x[i][j] == 1){
					img.Set(i, j, color.Black)
				}else{
					img.Set(i, j, color.White)
				}
			}
		}
		anim.Delay = append(anim.Delay, delay)
		anim.Image = append(anim.Image, img)
		x, E = simulation(x, E)
	}
	gif.EncodeAll(out, &anim)
	
	defer out.Close()
}
