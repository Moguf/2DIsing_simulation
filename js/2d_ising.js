onload = function() {
    draw();
};
var SIZE = 25;
function imshow(mat,ctx){
    for(var i = 0; i < SIZE ; i++){
        for(var j = 0; j < SIZE ; j++){
            ctx.rect(20*i,20*j,20*(i+1),20*(j+1));
            if(mat[i][j] == 1){
                ctx.fillStyle = "rgb(0,0,0)";
                ctx.fillRect(20*i,20*j,20,20);
            }
        }
    }
    ctx.stroke();
}

function draw(){
    var canvas = document.getElementById('canvas');

    var ctx = canvas.getContext('2d');

    mat = simulate();
    imshow(mat,ctx);

}
    
function simulate(){
    var mat = new Array(SIZE);

    for(var i = 0; i < SIZE ; i++){
        mat[i] = new Array(SIZE);
        for(var j = 0; j < SIZE ; j++){
            tmp = Math.random();
            if(tmp <= 0.5)
                mat[i][j] = 1;
            else
                mat[i][j] = 0;
        }
    }
    return mat;
}
