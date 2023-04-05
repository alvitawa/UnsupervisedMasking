FROM python:3.9 as global

# non interactive frontend for locales
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -q && apt-get install -qy  \
    texlive-full  \
    gnuplot  \
    make  \
    git

RUN apt install -y default-jre

RUN mkdir -p /app && mkdir -p /app/plots
WORKDIR /app


FROM global as resnet_model_size_plot
COPY ./plots/resnet_model_size /app/plots/resnet_model_size
RUN cd /app/plots/resnet_model_size && arara resnet_model_size.tex || (cat resnet_model_size.log && exit 1)
RUN ls /app/plots/resnet_model_size

#FROM global as thesis
#COPY --from=resnet_model_size_plot /app/plots/resnet_model_size/resnet_model_size.pdf /app/plots/resnet_model_size/resnet_model_size.pdf
#COPY ./title-page-ai.tex ./title-page-ai.tex
#COPY ./uvalogo_regular_p_nl.eps ./uvalogo_regular_p_nl.eps
#COPY ./bibentries.bib ./bibentries.bib
##RUN arara title-page-ai.tex || (cat title-page-ai.log && exit 1)
#RUN pdflatex title-page-ai.tex #|| (cat title-page-ai.log && exit 1)
#RUN biber title-page-ai #|| (cat title-page-ai.log && exit 1)
#RUN pdflatex title-page-ai.tex #|| (cat title-page-ai.log && exit 1)
##RUN pdflatex title-page-ai.tex #|| (cat title-page-ai.log && exit 1)

FROM global as neurips
COPY ./plots/part2_accuracy/vertical ./plots/part2_accuracy/vertical
COPY ./plots/part2_accuracy/full ./plots/part2_accuracy/full
COPY ./neurips_2022.sty ./neurips_2022.sty
COPY ./neurips_2022.tex ./neurips_2022.tex
COPY ./bibentries.bib ./bibentries.bib
RUN pdflatex neurips_2022.tex
RUN #biber neurips_2022
RUN pdflatex neurips_2022.tex

FROM global as link
COPY --from=thesis /app/title-page-ai.pdf /app/title-page-ai.pdf
COPY --from=neurips /app/neurips_2022.pdf /app/neurips_2022.pdf