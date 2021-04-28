close all
clearvars
clc

user = 1;
e = load(['/Users/saeedkazemi/Documents/reproducible-research/data/Step Scan Dataset/Barefoot/Participant',num2str(user),'.mat']);
e = struct2cell(e);
I = e{1,1};

binary_treshold = 2;
for i =600:1000%:1210
    se = ones(5,5);
    Co = (double(I(:,:,i)));%imbinarize,binary_treshold);
    %figure;imshow(Co,[])
    C = imdilate(Co,se);
    %figure;imshow(C,[])
    C = imbinarize(C,binary_treshold);
    %figure;imshow(C,[])
    
    
    S{i} = regionprops(C...
        , 'Centroid','MajorAxisLength','MinorAxisLength',...
        'Area', 'BoundingBox', 'Image');
    

    k=1;
    while k <= size(S{i},1) && ~isempty(S{i})
        if S{i}(k).Area<50
            S{i}(k)=[];
            k=1;
            continue
        end
        k = k+1;
    end
    if ~isempty(S{i})
        S{i}(1).group = 1;
        S{i}(1).box = S{i}(1).BoundingBox;
        for k=2:size(S{i},1)
            if norm(S{i}(1).Centroid-S{i}(k).Centroid)<40
                S{i}(k).group = 1;
                S{i}(1).box(1,1) = min(S{i}(1).box(1,1),S{i}(k).BoundingBox(1,1));
                S{i}(1).box(1,2) = min(S{i}(1).box(1,2),S{i}(k).BoundingBox(1,2));
                
                
                if S{i}(1).box(1,2) >= S{i}(k).BoundingBox(1,2)
                    A=S{i}(k).BoundingBox(1,3);
                    B=S{i}(k).BoundingBox(1,4);
                else
                    A=S{i}(1).BoundingBox(1,3);
                    B=S{i}(1).BoundingBox(1,4);
                end
                    
                    
                S{i}(1).box(1,3) = A...
                    +(max(S{i}(1).box(1,1),S{i}(k).BoundingBox(1,1))-min(S{i}(1).box(1,1),S{i}(k).BoundingBox(1,1)));
                S{i}(1).box(1,4) = B...
                    +(max(S{i}(1).box(1,2),S{i}(k).BoundingBox(1,2))-min(S{i}(1).box(1,2),S{i}(k).BoundingBox(1,2)));
            else
                S{i}(k).group = 2;
                if isempty(S{i}(2).box)
                    S{i}(2).box = S{i}(k).BoundingBox;
                else
                    S{i}(2).box(1,1) = min(S{i}(2).box(1,1),S{i}(k).BoundingBox(1,1));
                    S{i}(2).box(1,2) = min(S{i}(2).box(1,2),S{i}(k).BoundingBox(1,2));

                    if S{i}(2).box(1,2) >= S{i}(k).BoundingBox(1,2)
                        A=S{i}(k).BoundingBox(1,3);
                        B=S{i}(k).BoundingBox(1,4);
                    else
                        A=S{i}(2).BoundingBox(1,3);
                        B=S{i}(2).BoundingBox(1,4);
                    end
                    S{i}(2).box(1,3) = A...
                        +(max(S{i}(2).box(1,1),S{i}(k).BoundingBox(1,1))-min(S{i}(2).box(1,1),S{i}(k).BoundingBox(1,1)));
                    S{i}(2).box(1,4) = B...
                        +(max(S{i}(2).box(1,2),S{i}(k).BoundingBox(1,2))-min(S{i}(2).box(1,2),S{i}(k).BoundingBox(1,2)));
            
                    
                end
            end
        end
    end    
    
    
    J = insertText(double(I(:,:,i)), [1 1 ], ['subject = ',num2str(user)]);
    imshow(J, [])

    if ~isempty(S{i})

        
        hold on
        rectangle('Position',S{i}(1).box,...
              'Curvature',0.5,...
             'LineWidth',2,'LineStyle','--', 'EdgeColor', 'r')
        if size(S{i},1)>=2 && ~isempty(S{i}(2).box)
            hold on
            rectangle('Position',S{i}(2).box,...
                  'Curvature',0.5,...
                 'LineWidth',2,'LineStyle','--', 'EdgeColor', 'r')
        end
        hold off

        
    %else
        %B = insertText(J,[320 50],'Not Found','Font','LucidaBrightRegular','BoxColor','w');
        %imshow(B, [])
    end
    
    title(['image number: ', num2str(i)]) 

    pause(0.001);
    %close(gcf)
    
end