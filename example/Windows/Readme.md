#Come compilare
1) Creare la cartella install-Lib Attenzione quando eseguo il comando install() di Cmake questa cartella ha proprietà solo lettura 
cambiare dalle proprietà

2) Compilare la libreria ai4prod e installarla in install-Lib(di default è giù così)

# Compilare con

   $ mkdir build
   $ cd build
   $ cmake -G "Visual Studio 15 2017" -A x64 ..
   $ cmake --build . --config Release