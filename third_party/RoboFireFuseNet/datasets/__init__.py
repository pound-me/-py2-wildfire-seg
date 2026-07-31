"""Local package marker for the vendored RoboFireFuseNet dataset modules.

The upstream snapshot omits this file.  Keeping the directory as an explicit
package prevents an installed third-party package also named ``datasets`` from
shadowing ``wildfire.py`` on remote training hosts.
"""
