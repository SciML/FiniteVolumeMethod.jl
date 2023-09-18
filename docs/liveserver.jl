const repo_root = dirname(@__DIR__)
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
import LiveServer 
withenv("LIVESERVER_ACTIVE" => "true") do
    LiveServer.servedocs(;
        launch_browser=true,
    )
end

