<!--
SPDX-FileCopyrightText: gb-dispatch-model contributors
SPDX-License-Identifier: CC0-1.0
-->

# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a Vulnerability

To report a vulnerability, the preferred method is to use the GitHub ["Report a vulnerability" button](https://github.com/open-energy-transition/gb-dispatch-model/security/advisories/new) under the "Security and quality" tab.
This will create a private communication channel between the reporter and the repository maintainers.

If you are absolutely unable to or have strong reasons not to use GitHub's vulnerability reporting workflow, please reach out to the OET security team: security@openenergytransition<!--.example-->.org.

A lead maintainer or security team member will acknowledge your report within 2 business days, and will follow up with next steps shortly after.
The security team will endeavor to keep you informed of the progress towards a fix and full announcement, and may ask for additional information or guidance.

Please report security bugs in third-party modules to the person or team maintaining that module.

## Disclosure Policy

When the team receives a security bug report, they will assign it to a primary handler.
This person will coordinate the fix and release process, involving the following steps:

- Confirm the problem and determine the affected versions.
- Audit code to find any potential similar problems.
- Prepare fixes according to the patching targets below.

Security fixes are given priority and might be enough to cause a new version to be released.

## Patching Targets

Time-to-patch targets from verification of the report:

| Severity | Target |
| -------- | ------ |
| Critical | 48 hours |
| High | 14 days |
| Medium | 60 days |
| Low | Best effort / next scheduled release |

## End of Life

When this project is no longer maintained, the repository will be archived on GitHub and a notice added stating that security updates have ceased.
