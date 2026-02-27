FROM julia:1.10-bookworm

RUN useradd -m -u 1000 user

ENV JULIA_DEPOT_PATH=/opt/julia-depot
RUN mkdir -p $JULIA_DEPOT_PATH && chmod 777 $JULIA_DEPOT_PATH

WORKDIR /home/user/app

COPY Project.toml .
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

COPY . .

RUN chown -R user:user /home/user/app

USER user

EXPOSE 7860

CMD ["julia", "--project=/home/user/app", "/home/user/app/server.jl"]
