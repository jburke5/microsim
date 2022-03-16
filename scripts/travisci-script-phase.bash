#!/bin/bash
set -ev

# run tests in quick mode for commit pushes to any branch except `master`
if [[ "$TRAVIS_EVENT_TYPE" == 'push' && "$TRAVIS_TAG" == '' && "$TRAVIS_BRANCH" != 'master' ]];
then
    MICROSIM_TEST_QUICKMODE=true poetry run ./scripts/test.py
else
    poetry run ./scripts/test.py
fi
