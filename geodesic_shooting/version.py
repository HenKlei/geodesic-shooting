__version__ = '0.1.0'
try:
    import git
    try:
        repo = git.Repo(search_parent_directories=True)
        __version__ = repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        pass
except ModuleNotFoundError:
    pass
