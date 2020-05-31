[x,y] = meshgrid(0:1/69:1,0:1/69:1);
[X,Y] = meshgrid(0:1/24:1,0:1/24:1);
startx = reshape(X,[1,25*25]);
starty = reshape(Y,[1,25*25]);
%starty = ones(size(startx))/2;

u = table2array(velpre(:,3));
u = reshape(u,[70,70]);
v = table2array(velpre(:,4));
v = reshape(v,[70,70]);
streamline(x,y,u,v,startx,starty);