try:
    import git
    repo = git.Repo(search_parent_directories=True)
    __version__ = repo.head.object.hexsha
except (ModuleNotFoundError, git.exc.InvalidGitRepositoryError):
    __version__ = '0.1.0'
