library(shiny)
library(sf)
library(tidyverse)
library(viridis)
library(shinythemes)

options(shiny.host = "0.0.0.0")
options(shiny.port = 8180)

source('R/script.R')

ui_about <-fluidPage(
  h2("Spatial analysis of DRC artisanal mining sites data"),
  br(),
  br(),
  p("The aim of this work is to present insights from collected data of artisanal mining sites in DR Congo using commonly used spatial techniques. Each method is brievly presented."),
  p("We will work with IPIS' artisanal mining sites spatial data and territories data of DRC. Artisanal mining sites data can be download", a(href="https://github.com/jorisvandenbossche/geopandas-tutorial", "here.")),
  p("Author: Trésor DJONGA"),
  p("Contact: tresordjonga@gmail.com")


)

ui_data_proccessing <-fluidPage(
  
  h2("Pre-proccessing and vizualisation"),
  
  HTML('
  <p> For IPIS data, We worked with the following variables:</p>
  <ol >
    <li> Internal IPIS identifiant of artisanal mining sites</li>
    <li> Latest visited of artisanal mining sites</li>
    <li> Number of all peoples working in the in the artisanal mining sites</li>
    <li> Is there one of 3T mines (Cassiterite, Wolframite, Coltan)?</li>
    <li> Is there à gold?</li>
    <li> Is there a state or not state army group</li>
</ol>
    <p>All raw with missing values in each of those columns were deleted.</p>
    <p>We kept artisanal mining sites with number of workers less then 1000. Some artisanal mining sites were visited several time, so we will kept the latest visit.Finally, we will work at the provincial and territorial level, the data were aggregated respectively.</p>   '),
  
  
  
  fluidRow(class = "toprow",
           fluidRow (class = "filters",
                     
                     column(6,
                           
                            selectInput("province", "Region", choices = regions, selected = 'Sud-Kivu')
                     ),
                     
                     column(6,
                            
                            selectInput("mines", "Indicator", choices = c("Presence of armed group" = "SumPresence", "Gold" = "sumIs_gold_mine", "Workers in artisanal mine" = "sumWorkers_numb","3T mines (Cassiterite, Wolframite, Coltan)"= "sumIs_3t_mine" )) # Sort options alphabetically
                            
                     )
           )
  ),
  
  fluidRow ( 
    
    column(6, class = "map",
           
           plotOutput("map1")
    ),
    
    column(6, class = "table1",
           
           plotOutput("bar1")
    )
    
  ),
  
  fluidRow (class = "table",
          
            DT::DTOutput("table1")
  )
  
  )

ui_autocorrelation <-fluidPage(
  
  h2('Spatial autocorrelation'),
  withMathJax(),
  helpText("
The spatial autocorrelation will help us to undestand if there is a variation of our variable of interest across the space. We will discover if territories with artisanal mining which contains 3T and gold mines are close each other or if they are randomnly distributed across DRC. We will also see if territories with artisanal mining site with higher mean number are close each other.

To measure the spatial autocorrelation, we will use Moran's coefficient  denoted as I which is the extension of Pearson’s correlation denoted  \\(\\rho\\). Recall that:
$$\\rho(x,y) = \\frac{\\sum_{i=1}^{n}(x_{i} - \\bar{x}) (y_{i} - \\bar{y}) }{\\sqrt {\\sum_{i=1}^{n} (x_{i} - \\bar{x})^{2}   (y_{i} - \\bar{y})^{2}}}$$

where \\(x = (x_{1},\\ldots,x_{n}) \\), \\( x = (y_{1},\\ldots,y_{n}) \\), \\( \\bar{x}\\) sample mean of \\( x\\) and \\( \\bar{y}\\) sample mean of \\( y\\).

For a single variable, say \\( x \\), \\( I\\) will measure
whether \\(x_i \\) and \\(x_j \\), with \\(i \\neq j \\), are associated, in other words the correlation of \\(x \\) with herself spatially. \\(x \\) may be 3T mines, mean number of workers,...
Waldo Tober’s first law of geography is about “Everything is related to everything else, but near things are more related than distant things.” So logically expect that close observations are more likely to be similar with \\(x\\) than those far apart. Moran introduced weight \\( \\omega_{ij}\\) to take into account the nearest spatial objects (territories here) \\(x_i\\) and \\(x_j\\). Moran I’s formula is given by:
$$I = \\frac{n\\sum_{i=1}^{n}\\sum_{j=1}^{n} \\omega_{ij} (x_{i} - \\bar{x}) (x_{j} - \\bar{y}) }{ \\Omega\\sum_{i=1}^{n} (x_{i} - \\bar{x})^{2} }$$
We set \\( \\omega_{ii} = 0\\) and \\(\\Omega = \\sum_{i=1}^{n}\\sum_{j=1}^{n} \\omega_{ij}\\).

Closest territories will have weight 1 and O otherwise. \\(w_{ij}\\) can be taken equal to \\( \\frac{1}{d_{ij}}\\) where \\( d_{ij}\\) is the distance between \\(x_i\\) and \\(x_j\\). Moran I’s observed value will be compared to $$I_{0} = \\frac{-1}{n-1}$$."),
  
  helpText("1. If \\(I > I_0 \\) then we observe significantly positive autocorrelation.Similar values tend to be nearer together; "),
  helpText("2. If \\( I < I_0 \\) then we observe significantly negative autocorrelation. Similar values tend to be furtherest each other."),
  
  p("We start by finding neighbours for each DRC's territories."),
  verbatimTextOutput('neighbour'),
  
  p('The links between neighbours visualises  distribution across space.'),
  plotOutput('graphNeighbour'),
  
  helpText('A spatial neighborhood matrix \\( W \\)  defines a neighborhood structure over the
   entire study region, and its elements can be viewed as weights. The \\( (i, j) \\) th element of \\( W\\), denoted by \\(w_{ij} \\), spatially connects areas \\( i\\) and \\( j\\) in some fashion.'),
  plotOutput('weihtedMatrix'),
  
  HTML('
    <p>The Moran’s correlation score lies between -1 and 1.</p>
    <ol >
      <li>The 1 determines perfect postive spatial autocorrelation, simular values are grouped or clustered together </li>
      <li>The 0 indicates randomnly distributed of values</li>
      <li> -1 represents negative spatial autocorrelation (so dissimilar values are next to each other)</li>
    </ol>
    <p>The Moran’s correlation score lies between -1 and 1.</p>
    <p>The neighbour data need to be converted to a listw object.</p>
       '),
  verbatimTextOutput('moran1'),
  
  p("Global Moran’s test for mean number of workers across territories."),
  p("The moran’s is significant at level 5%. the p-value is 0.03166. Moran’s statistic is 0.19553943 greater than the expectation -0.02941176, we can therefore confirm that mean number of workers across territories is positively autocorrelated in DRC. Territories with similar values of mean number of workers are grouped"),
  verbatimTextOutput('moran2'),
  
  p("Global Moran’s test for sumIs_gold_mine across territories."),
  p("The moran’s is significant at level 5%. the p-value is 0.02284. Moran’s statistic is 0.03166 greater than the expectation -0.02941176 , we can therefore confirm that number of artisanal mining site containing gold across territories is positively autocorrelated in DRC. Territories with similar values of number gold in artisanal mining are grouped."),
  verbatimTextOutput('moran3'),
  
  p("Global Moran’s test for 3T mines across territories."),
  p("The moran’s is significant. the p-value is 9.567e-05. Moran’s statistic is 0.42712496 greater than the expectation -0.02941176 , we can therefore confirm that number of artisanal mining site containing 3T mines across territories is positively autocorrelated in DRC. Territories with similar values of number 3T mines in artisanal mining are grouped."),
  verbatimTextOutput('moran4'),
  
  p("Global Moran’s test for sumPresence across territories."),
  p("The moran’s is significant at level 5%. the p-value is 0.02981. Moran’s statistic is 0.42712496 greater than the expectation 0.18889882 , we can therefore confirm that number of artisanal mining site containing no state group army accros territories is positively autocorrelated in DRC. Territories with similar values of number no state group army in artisanal mining are grouped."),
  verbatimTextOutput('moran5'),
  
  p("All our variables of interest are positive autocorrelated across DRC’s territories. The previous graphics shown territories are clustered according the considered variable. Each province in DRC contains specific mines. Territories containing gold are the most clustered. We observed also the lack of no state group army in territories for the former Katanga provinces.")
)


ui_density_estimation <-fluidPage(
  h2('Spatial kernel density estimation'),
  withMathJax(),
  helpText("We are interested to cover a land with high density of artisanal mining sites in DRC. Kernel density is one of the most spatial method used for points pattern analysis and it produces a hot spots maps. The idea behind this method is from a set of discrete points \\( x_{1},\\ldots, x_{i},\\ldots,x_{n} \\) to produce a continuous density map or raster \\( \\hat{f}(x)\\) in which each pixel represents a density value based on the number of points within a given distance window width.
      
Given \\( x_{1},\\ldots,x_{i},\\ldots,x_{n} \\), \\( x_{i} \\in \\mathbf{R}^{d} \\), \\( \\hat{f}(x) \\) is given by:
$$\\hat{f}(x) = \\frac{1}{nh^{d}} \\sum_{i=1}^{n} \\mathbf{K}\\{\\frac{(x-x_{i})}{h}\\}$$
  
    
    where  \\(h \\) is the bandwidth , \\( \\mathbf{K} \\) is the Kernel function and \\( (x-x_{i}) \\) represents the Euclidian distance between each point \\( i \\) and the location where the density estimator is worked out.
    
    \\newline
    
   For hotspot analysis, the case mostly used are : 1. \\( d = 2\\), 2. The bivariate normal kernel \\( \\mathbf{K}(x) = \\frac{1}{2\\pi} \\exp(-\\frac{1}{2}x^{t}x) \\), 3.The reference bandwith \\( h = \\left(0.5 (sd_{x_{1}} + sd_{x_{2}})\\right)n^{−1/6} \\) where \\( sd_{x_{1}} \\) and \\( sd_{x_{2}}\\) are respectively standard deviation of the coordinates \\( x_1 \\) and \\( x_2 \\).,
  
    
           "),
  p("We will work artisanal mining sites dataIPIS"),
  
  HTML('
<b>As you can bellow, and which covers territories of Walikale, Masisi, Shabunda, Kaba, kale contains the highest density of artisanal mining sites. Oicha, lubero and Irumu</b>
'),
  
  plotOutput('kernelRDC'),
  plotOutput('kernelPixelRDC'),
  plotOutput('hotspot')
  
)

ui_spatial_interpolation_idw <-fluidPage(
  h2('Spatial interpolation: Inverse Distance Weighted'),
  withMathJax(),
  
  helpText("Given a set of measures \\( x_{1} ,\\ldots,x_{i},\\ldots,x_{n} \\) (mean number of workers, number of non state group army,..) in different locations(artisanal mining sites), the aim of interpolation is to predict value \\( x_{j} \\) at location where no measurements have been made. We encounter several spatial interpolation techniques in the literature,for example:"),
  helpText("1.Thiessen Polygon: Assigns interpolated value equal to the value found at the nearest sample location. Only one point used, the nearest. Accuracy depends largely on sampling density."),
  helpText("2.Inverse Distance Weighted (IDW): Assigns interpolated value after using the distance and values to nearby know points.Weight of each sample point is an inverse proportion to the distance. The further away the point, the less is the weight.The value \\( x_j\\) is given by:
$$x_{j} = \\frac{\\sum_{ i =1}^{n}x_{i}\\omega_{ij}}{\\sum_{ i =1}^{n}\\omega_{ij}}.$$"),
  helpText("Where \\( \\omega_{ij} = \\frac{1}{d_{ij}^{\\alpha}} \\), \\( d_{ij} \\) is the distance between \\( x_{j} \\) and \\( x_{i} \\) ,and \\( \\alpha \\) a user defined coefficient."),
  p("We will run the the model on DRC’s territories without katanga and then test with artisanal mining sites of katanga."),
  
  p('Prediction mean number of workers in artisanal site'),
  verbatimTextOutput('idw'),
  
  
  p("Prediction on Katanga artisanal mining sites"),
  verbatimTextOutput('idwPred'),
  
  p("Root mean square error"),
  verbatimTextOutput('idwRMSE'),
  
  p("chlororopheth  prediction on katanga territories"),
  plotOutput('idwGraph')
  
  )

ui_weighted_regression <-fluidPage(
  h2('Geographically weighted regression'),
  withMathJax(),
  
  helpText("Geographically weighted regression(GWR) is the extension of regression where the stationary process is not assumed. If we consider globally our data,the prediction of mean number of workers can be obtained using:
$$y_{i} = \\beta_{0} + \\sum_{j =1}^{m} \\beta_{j}x_{ji} + \\epsilon_{i}.$$"),
  
  helpText("where \\( \\beta_{0} \\) the intercept, \\( \\beta = (\\beta_{1},\\ldots,\\beta_{m}) \\) regression model coefficient, \\( \\epsilon \\) residual and \\( x_{ji} \\) independant variables(is_3t_mine, is_gold_mine, presence). Recall that the estimator of \\( \\beta \\) is:
$$\\hat{\\beta} = (X^{t}X)^{-1}X^{t}Y.$$"),
  
  helpText("The coefficient \\( \\hat{\\beta} \\) is the same over the space. However the administration is different across DRC territories, influence of army  group is different because the government in Kinshasa does not control the whole country,..."),
  
  helpText("The GWR take into account this spatial variability and autocorrelation, and introduces the weight $\\omega$. So,
$$y_{i} = \\beta_{0}(i) + \\sum_{j =1}^{m} \\beta_{j}(i)x_{ji} + \\epsilon_{i}$$ The estimator is:
$$\\hat{\\beta}(i) = (X^{t}W(i) X)^{-1}X^{t}W(i)Y.$$"),
  
  helpText("where \\( W(I) \\) is a weight matrix diagonal for specific location \\( I \\) so that nearby observation are given greather weight than further away.
     $$W(i) = \\begin{pmatrix}
     w_{i1} & 0 &\\ldots & 0\\\\
     0 & w_{i1} & \\ldots & 0\\\\ 
    \\ldots & \\ldots & \\ldots & \\ldots\\\\
    0 & 0 &\\ldots & w_{im}
    \\end{pmatrix}$$"),
  
  helpText("\\( w_{im} \\) is the weight given to point \\( m \\) for the estimate of the local parameters at
location \\( i\\). There are a range for weighting calibration, for instance:
$$w_{ij} = \\exp \\{-\\frac{1}{2} (\\frac{d_{ij}}{h})^{2}\\}$$ or $$w_{ij}=
    \\begin{cases}
      ( 1-\\frac{d_{ij}^{2}}{h^{2}})^{2}, & \\text{if j is one of the Nth nearest neighbours of i}\\ a=1 \\\\ 
      0, & \\text{otherwise}
    \\end{cases}$$
           where \\( d_{ij} \\) is the distance between locations \\( i \\) and \\( j\\) and \\( h \\)is the bandwidth. "
           ),
  
  helpText("GWR is used as a data exploration technique that allows to identify if locally weighted regression coefficients may vary across the study area.\
We first run global regression model with territories data."),
  
  verbatimTextOutput('regModel'),
  
  helpText('The global model is not significant at 5% level, p-value is 0.474, R squared explains only 7% of variance and only the intercept is significant, all independent variables are not significant.'),
  helpText('To run the GWR regression, we need first to calculate a kernel bandwidth.'),
  
  verbatimTextOutput('gwr'),
  HTML('
<b>Quasi globally r squared is 10% and locally coefficents remain close to the global coefficient.</b>
'),
  
  plotOutput('graphGWR')
)

ui_clustering <-fluidPage(
  h2('Clustering spatial data'),
  withMathJax(),
  
  helpText('The clustering algorithm is applied only to non spatial attributes and spatial units are assigned to the cluster of their non spatial attributes. We present bellow Kmeans, one of the simplest clustering algorithm.

Let \\( X = \\{x_1, . . . , x_n\\} \\) be a set of \\( n \\) spatial units, each with $d$ non spatial attributes.

The Kmeans problem aims to find a set of \\( k \\) groups of spatial  which minimizes the function $$f(M) = \\sum_{x \\in X} \\min_{\\mu \\in M}|| x- \\mu||^{2}_{2}$$
where \\( M =\\{\\mu_{1},\\ldots,\\mu_{k}\\} \\), \\( \\mu_{k} \\) is the centroid and \\( k\\) the number of the clusters and the input parameter. The clustering algorithm is applied only to non spatial attributes and spatial units are assigned to the cluster of their non spatial attributes. We present bellow Kmeans, one of the simplest clustering algorithm.

Let \\( X = \\{x_1, . . . , x_n\\} \\) be a set of \\( n \\) spatial units, each with $d$ non spatial attributes.

The Kmeans problem aims to find a set of \\( k \\) groups of spatial  which minimizes the function $$f(M) = \\sum_{x \\in X} \\min_{\\mu \\in M}|| x- \\mu||^{2}_{2}$$
where \\( M =\\{\\mu_{1},\\ldots,\\mu_{k}\\} \\), \\( \\mu_{k} \\) is the centroid and \\( k \\) the number of the clusters and the input parameter. '),
  
  helpText('Step of the Kmeans clustering'),
  helpText('1.Choose \\( k \\)  initial centroids \\( µ_1,\\ldots, µ_k \\) randomnly from the set \\( X \\)'),
  helpText('2.For each point \\( x \\in X \\), find the closest centroid \\( µ_i\\) and add \\( x\\) to a set \\( S_i\\)'),
  helpText('3.For \\( i = 1,\\ldots k \\), set \\( µ_i \\) to be the centroid of the points in \\( S_i\\)'),
  helpText('4. Repeat steps 2 and 3 until all centroids do not change or \\( f(M) \\) be less a given precision'),
  
  HTML(' 
  <center>
     <b>Regionalization.</b>
  </center>   
'),
  
  helpText("Regionalization called also constrained spatial clustering aims to aggregate the geographic units into a number of spatially contiguous groups while optimizing an objective function, which is normally
a measure of the attribute similarity in each group. REDCAP (GUO 2008) and SKATER are the most used regionalization techniques.\

Running Kmeans with territories data for 4 clusters.The result's of Kmeans depends of randomly chosen centroids, so we ask R to run the model 50 times and choose the best. Data are also scaled."),
  
  verbatimTextOutput('geoClustering'),
  
  HTML(' 
  <center>
     <b>Representation of territories with their clusters.</b>
  </center>   
'),
  
  plotOutput('graphClusters'),
  

  HTML(' 
  <center class>
     <b class = "titleb">Table of territories with their clusters.</b>
  </center>   
'),
  
  DT::DTOutput("clusterRegions"),
    
  p('Carateristics of cluster'),
  DT::DTOutput("clusters")
  
  )




ui <- fluidPage( theme = "styles.css",
  
  
  navlistPanel(
   "" ,well = FALSE, 
   tabPanel("About", ui_about 
   ),
  tabPanel("Data loading and pre-processing", ui_data_proccessing
  ),
  
  tabPanel("Spatial autocorrelation", ui_autocorrelation 
  ),
  
  tabPanel("Spatial kernel density estimation", ui_density_estimation
  ),
  
  tabPanel("Spatial interpolation:Inverse Distance Weighted",ui_spatial_interpolation_idw
  ),
  

  
  tabPanel("Geographically weighted regression", ui_weighted_regression
  ),
  
  tabPanel("Clustering spatial data", ui_clustering
  )
  
)
)


server <- function(input, output) {
  
  #-----------------Preparation-------------------------
  data_region <-reactive({
    temp <- territories
    if (input$province != "All"){
      temp <- territories %>% filter( Province == input$province)
    }
    temp
  })
  
  df_temp <- reactive({
      df<- data.frame (
       kpi = c( "Gold", "Armed grouped", "3T Mine", "Workers(in thousand)"),
       value = c(sum(data_region()$sumIs_gold_mine, na.rm=T), sum(data_region()$SumPresence, na.rm=T), sum(data_region()$sumIs_3t_mine, na.rm=T), sum(data_region()$sumWorkers_numb, na.rm=T)/1000)
     )
      df
 })
  
  output$map1 <-renderPlot({
    gg <- ggplot(data_region()) + geom_sf(aes(fill = .data[[input$mines]])) + scale_fill_viridis() + geom_sf_label(aes(label = NOM)) + geom_sf_label(aes(label = NOM)) + labs(fill = ' ')+ theme_bw() + theme(axis.title.x = element_blank(),axis.title.y = element_blank())
    print(gg)
  })
  
  output$table1 <- DT::renderDT({
    df <-data_region() %>% filter(!is.na(.data[[input$mines]])) %>% dplyr::select(-meanWorkers_numb, -TYPE, -SCE_SEM, -SCE_GEO,-MODIF, -ORIGINE, -Pcode, -Province, -idCluster) %>% st_drop_geometry()
    datatable(df)  %>% formatStyle("sumWorkers_numb", backgroundColor = styleInterval(getQuantile(df$sumWorkers_numb), getColor(df$sumWorkers_numb))) %>% 
         formatStyle("sumIs_3t_mine", backgroundColor = styleInterval(getQuantile(df$sumIs_3t_mine), getColor(df$sumIs_3t_mine))) %>%
         formatStyle("sumIs_gold_mine", backgroundColor = styleInterval(getQuantile(df$sumIs_gold_mine), getColor(df$sumIs_gold_mine))) %>%
         formatStyle("SumPresence", backgroundColor = styleInterval(getQuantile(df$SumPresence), getColor(df$SumPresence)))
  })
  
  output$bar1 <-renderPlot({
    
    gg <- ggplot(df_temp()) + aes(x = kpi, y = value) + geom_col(fill = "#299b3b", width = .5 ) + geom_text(aes( label= value), hjust= 0.5, vjust = 1.4, size= 4.0, color= "#ce522e") + 
        theme_bw() + theme(axis.title.x = element_blank(),axis.title.y = element_blank())
    print(gg)
  })
  
  #spatial autocorrelation
  
  output$neighbour <-renderPrint({ 
     neighbours
    })
  
  output$graphNeighbour <-renderPlot({
    plot(st_geometry(territories), border = "green")
    plot.nb(neighbours, st_geometry(territories), add = TRUE)
  })
  
  output$weihtedMatrix <-renderPlot({
    nbw <- spdep::nb2listw(neighbours, style = "W", zero.policy = TRUE)
    m1 <- listw2mat(nbw)
    lattice::levelplot(t(m1),
                       scales = list(y = list(at = c(10, 20, 30, 40),
                                              labels = c(10, 20, 30, 40))))
  })
  
  output$moran1 <-renderPrint({ 
    print(listw, zero.policy=TRUE)
  })
  
  output$moran2 <-renderPrint({ 
    moran.test(territories$meanWorkers_numb, listw, zero.policy=TRUE)
  })
  
  output$moran3 <-renderPrint({ 
    moran.test(territories$sumIs_gold_mine, listw, zero.policy=TRUE)
  })
  
  output$moran4 <-renderPrint({ 
    moran.test(territories$sumIs_3t_mine, listw, zero.policy=TRUE)
  })
  
  output$moran5 <-renderPrint({ 
    moran.test(territories$SumPresence, listw, zero.policy=TRUE)
  })
  
  #----Desnity estimation--------------------------------------------------
  output$kernelRDC <-renderPlot({ 
    image(kernelDRC)
    })
  
  output$kernelPixelRDC <-renderPlot({
    plot(pixelkernel)
  })
  
  output$hotspot <- renderPlot({
    gg <- ggplot(data =  as.data.frame(kde, xy = TRUE)) +
      geom_raster(aes(x = x, y = y, fill = Hotspot)) +
      geom_sf(data = territories,  aes(label = NOM)) + scale_fill_viridis() 
    
    print(gg)
  })
  
  #spatial interpolation IDW
  output$idw <-renderPrint({
    idw
  })
  
  output$idwPred <-renderPrint({
    idw$var1.pred
  })
  
  output$idwRMSE <-renderPrint({
    RMSE(testingSet$meanWorkers_numb,idw$var1.pred)
  })
  
  output$idwGraph <- renderPlot({
    gg <-ggplot() + geom_sf(data = idw, aes(fill = var1.pred))  + theme_bw()
    print(gg)
    
  })
  
  #-----------------------Geographically weighted regression--------------------------
 output$regModel <-renderPrint({
   modelGlobal <- lm(territories$meanWorkers_numb ~ territories$sumIs_3t_mine+territories$sumIs_gold_mine +territories$SumPresence)
   summary(modelGlobal)
 })
  
  output$gwr <-renderPrint({
    gwrModel
  })
  
  output$graphGWR <-renderPlot({
    resultsLocal <-as.data.frame(gwrModel$SDF)
    gwrMap <- cbind(latlon, as.matrix(resultsLocal))
    
    gg<- ggplot(gwrMap) + geom_sf(aes(fill = localR2)) + theme_void()
    print(gg)
  })
  
  #--------------------Clustering----------------------------------------------
  
  output$geoClustering <-renderPrint({
    territoriesKmeans
  })
  
  output$graphClusters <- renderPlot({
    gg <- ggplot(territories) +
      geom_sf(aes(fill = as.factor(idCluster) )) +
      theme_bw() +
      labs(fill = 'Clusters')
    
    print(gg)
  })
  
  output$clusterRegions <-DT::renderDT ({
    territories %>% dplyr::select(NOM, idCluster) %>% st_drop_geometry()
  })
  
  output$clusters <-DT::renderDT ({
    df <-clusters
    datatable(df)  %>% formatStyle("meanWorkers_numb", backgroundColor = styleInterval(getQuantile(df$meanWorkers_numb), getColor(df$meanWorkers_numb))) %>% 
      formatStyle("sumIs_3t_mine", backgroundColor = styleInterval(getQuantile(df$sumIs_3t_mine), getColor(df$sumIs_3t_mine))) %>%
      formatStyle("sumIs_gold_mine", backgroundColor = styleInterval(getQuantile(df$sumIs_gold_mine), getColor(df$sumIs_gold_mine))) %>%
      formatStyle("SumPresence", backgroundColor = styleInterval(getQuantile(df$SumPresence), getColor(df$SumPresence))) %>%
      formatStyle("sumWorkers_numb", backgroundColor = styleInterval(getQuantile(df$sumWorkers_numb), getColor(df$sumWorkers_numb))) 
       
  })
  
}



shinyApp(ui = ui, server = server)
