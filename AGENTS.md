# Code Style:
1. Write unit-testable code.

2. Red/green TDD

3. Parametrize tests (i.e. cross product) 
   to keep them tight while exhaustively testing all cases

4. No defensive programming. 
   Do not worry about backwards compatibility within 
   the codebase as you make changes. 
   The project is in version 0.0.x.
   Do not document "what used to happen"

5. Refactor code. Survey code before 
   working and consider ways you can build 
   the feature or fix the bug by refactoring code,
   rather than just adding new code. 
   PRs should have a healthy mix of deleted
   LOC alongside new LOC. large diffs are a cost, not an accomplishment.

6. Constants, copy, formulas, config should be in their
   own files so that they are easy for humans to
   audit and review

8. All text (UI, docstrings, PR descriptions) should use
   Simplified Technical English (STE) with complete sentences.
   No metaphors, em-dashes, personifications, or aphorisms.

# Enforced & Precommit Checks:
1. The following should be locally enforced before a commit can be made and are tested in CI:
   - Ruff (formatting and style guide)
   - API documentaiton (Numpy style)
   - Dead code identificaiton (Vulture)

4. Additionally, the following are enforced in CI:
   - Codebase test coverage is > 95%

# Workflow:
1. Stack PRs for larger features. 

2. In general, PRs should be less than <1,000 lines of code 
   aside from tests

3. PRs should describe:
   - before/after
   - approach
   - alternatives considered, incl approaches taken in
     other projects or literature
   - all assumptions made: business logic, constants,
     semantic definitions or assumptions, decision points,
     etc.
    - Do not write e.g. "400 tests pass". Reviewers can see whether tests pass in the CI.
    - Try to keep PR descriptions short and terse.
   This software exists within a greater organization
   and reviewers will need to be aware of and 
   responsible for decisions made,
   so communicate these so that they can
   direct and support accordingly.

4. Only fix the main issue at hand. 
   If you find additional issues while working,
   log the issue in github and tag it with the subfeature