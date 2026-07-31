<!-- SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur> -->
<!-- SPDX-FileCopyrightText: Contributors to gb-dispatch-model <https://github.com/open-energy-transition/gb-dispatch-model> -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Installation {#installation_gb}

The subsequently described installation steps are demonstrated as shell commands.
To use them in your shell, copy them individually.

## Clone the Repository

First of all, clone the [gb-dispatch-model repository](https://github.com/open-energy-transition/gb-dispatch-model) using the version control system `git` in the command line.

```console
git clone https://github.com/open-energy-transition/gb-dispatch-model.git
```

## Create working environment

gb-dispatch-model relies on a set of other Python packages to function.
We manage these using [pixi](https://pixi.sh/latest/).
To install `pixi`, follow their [installation instructions](https://pixi.prefix.dev/latest/installation/).
You can also install `pixi` into a `conda` environment using the command line:

```console
conda install pixi
```

Once pixi is installed, you can run various tasks in the command line.
See our `Running` page for more information.

!!! note
    We don't currently support linux operating systems using ARM processors since certain packages, such as `PySCIPOpt`, require being built from source.

## Access API keys

For successful execution, you will need the ENTSO-E Transparency platform API key, stored in a `.env` file in your working directory.
To get this key, you will need to follow the [instructions provided by the transparency platform helpdesk](https://transparencyplatform.zendesk.com/hc/en-us/articles/12845911031188-How-to-get-security-token):

First create an [ENTSO-E transparency platform](https://transparency.entsoe.eu/) account.
Then, contact the ENTSO-E helpdesk by emailing `transparency@entsoe.eu` with the email subject `Restful API access` and email body being just the email address associated with your account.

Your `.env` file should then look like:

```console
ENTSO_E_API_KEY=<your-api-key-here>
```
