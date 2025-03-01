FROM rocker/shiny

RUN mkdir /home/ProjetDRCMining

RUN R -e "install.packages(c('tidyverse','readxl','tidyr','xlsx','shinydashboard','scales','DT','shinyjs', 'sf','lemon', 'viridis', 'shinythemes'))"

WORKDIR /home/ProjetDRCMining/
COPY app.R  app.R
ADD R R/
ADD www www/
ADD data data/

# Expose the application port
EXPOSE 8180

CMD Rscript /home/ProjetDRCMining/app.R

