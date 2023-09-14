const repo_root = dirname(@__DIR__)
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
import LiveServer
withenv("LIVESERVER_ACTIVE" => "true") do
    LiveServer.servedocs(;
        launch_browser=true,
        foldername=joinpath(repo_root, "docs"),
        include_dirs=[joinpath(repo_root, "src")],
        skip_dirs=[
            joinpath(repo_root, "docs/src/tutorials"),
            joinpath(repo_root, "docs/src/figures"),
            joinpath(repo_root, "docs/src/wyos"),
        ],
        include_files=[
            joinpath(repo_root, "docs/src/tutorials/overview.md"),
            joinpath(repo_root, "docs/src/wyos/overview.md"),
        ]
    )
end

