library(shiny)
library(sf)
library(tidyverse)
library(spdep)
library(geojsonio)
library(adehabitatHR)
library(gstat)
library(xts)
library(spgwr)
library(DT)



#----------------------------DATA PREPARATION-----------------------------------
dataIPIS <- st_read("data/cod_mines_curated_all_opendata_p_ipis.geojson")
ins_territoire <-read.csv('data/ins_territoire_drc.csv', sep = ';')
territories <-st_read("data/territoire", "Territoire")

crs_ips <- st_crs(dataIPIS)

dataIPIS <- dataIPIS %>%
  dplyr::select(pcode,visit_date, workers_numb,is_3t_mine,is_gold_mine, presence, visit_date) %>%
  filter(!is.na(visit_date) & !is.na(workers_numb)& !is.na(is_3t_mine)& !is.na(is_gold_mine)& !is.na(presence) ) %>%
  filter(workers_numb < 1000)%>%
  arrange(pcode, desc(visit_date)) %>%
  group_by(pcode) %>%
  slice(1)

territories <- territories %>%
  filter(TYPE=="Territoire") %>%
  mutate(NOM = ifelse(NOM== 'Oicha', 'Beni', NOM)) %>%
  left_join(ins_territoire, by = c("NOM" = "Territoire"))

#Reprojection territoire to dataIPIS' CRS.
territories <- st_transform(territories, crs_ips)

#join territories data with IPS
mines_data <- dataIPIS %>%
  dplyr::select(workers_numb, is_3t_mine, is_gold_mine, presence)  %>%
  st_join(territories,  join = st_intersects,  left = FALSE) %>%
  dplyr::select(workers_numb, is_3t_mine, is_gold_mine, presence, NOM) %>%
  st_drop_geometry() %>%
  as.data.frame()

#groupped data
mines_data_grouped <- mines_data  %>% 
  group_by(NOM) %>%
  summarise(meanWorkers_numb = mean(workers_numb),
            sumWorkers_numb = sum(workers_numb),
            sumIs_3t_mine =sum(is_3t_mine),
            sumIs_gold_mine = sum(is_gold_mine),
            SumPresence = sum(presence))

#Import grouped data to the territory dataset
territories <- territories %>%
  left_join (mines_data_grouped) 

regions <- c(unique(territories$Province), "All")
print(regions)

#--------------------------------------------------------
territories <-territories %>%
   filter( !is.na(meanWorkers_numb)& !is.na(sumIs_3t_mine)& !is.na(sumIs_gold_mine)& !is.na(SumPresence) )

neighbours <- poly2nb(territories, queen = TRUE)
listw <- nb2listw(neighbours, zero.policy=TRUE)


#-----------------------------Density estimation-------------------------

dataIPIS_sp <- geojson_read('data/cod_mines_curated_all_opendata_p_ipis.geojson', what='sp') 
dataIPIS_sp$ident <-rep("Hotspot",dim(dataIPIS_sp)[1])
kernelDRC = kernelUD(dataIPIS_sp[, "ident"]) 

pixelkernel<-estUDm2spixdf(kernelDRC)
#kde <- raster(pixelkernel)
#projection(kde) <- CRS(proj4string(dataIPIS_sp))

#------------------Spatial interpolation---------------------------------------
#Root mean square error for Model comparison.

RMSE <- function(observed, predicted) {
  sqrt(mean((predicted - observed)^2, na.rm=TRUE))
}

katanga <-c("Lubudi", "Mitwaba","Bukama", "Pweto", "Moba", "Manono", "Malemba-Nkulu", "Kalemie", "Nyunzu")

trainSet <-territories %>% filter(!(NOM %in% katanga))
testingSet <-territories %>% filter( NOM %in% katanga)

idw <- idw(trainSet$meanWorkers_numb ~ 1, locations = trainSet, newdata= testingSet)

#----------------Geographically weighted regression-----------------------------
latlon <-st_transform(territories, crs = "+proj=longlat +datum=WGS84")
sf_cent <- st_centroid(latlon)
st_coordinates(sf_cent)
GWRbandwidth <- gwr.sel(meanWorkers_numb ~ sumIs_3t_mine + sumIs_gold_mine + SumPresence, data=latlon, coords=st_coordinates(sf_cent), adapt =TRUE)

gwrModel = gwr(meanWorkers_numb ~ sumIs_3t_mine + 
                 sumIs_gold_mine + SumPresence, data=latlon, adapt=GWRbandwidth,coords=st_coordinates(sf_cent), hatmatrix=TRUE, se.fit=TRUE)

#----------------------------Clustering------------------------------------------------
tempData <-territories[, c('meanWorkers_numb', 'sumWorkers_numb', 'sumIs_3t_mine', 'sumIs_gold_mine', 'SumPresence')] %>% 
  st_drop_geometry()

territoriesKmeans<- kmeans(tempData, 4, nstart = 50)
territories$idCluster <-territoriesKmeans$cluster

clusters <- as.data.frame(aggregate(tempData, list(territories$idCluster), mean)) %>% mutate(across(c(meanWorkers_numb,sumWorkers_numb,sumIs_3t_mine,sumIs_gold_mine,SumPresence), round, 2))
#------------------------------styling---------------------------------
getQuantile <-function(col){
  brks <- quantile(col, probs = seq(.05, .95, .05), na.rm = TRUE)
  brks 
}

getColor <-function(col) {
  breaks <- getQuantile(col)
  clrs <- round(seq(255, 40, length.out = length(breaks) + 1), 0) %>%
    {paste0("rgb(255,", ., ",", ., ")")}
  
  clrs
}
