$pdf_mode = 5;
$postscript_mode = 0;
$dvi_mode = 0;
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error %O %S';

# Enable synctex for forward/inverse search
$pdflatex = "pdflatex -synctex=1 %O %S";

