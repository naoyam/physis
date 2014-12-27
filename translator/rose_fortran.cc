// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/rose_fortran.h"

#include <boost/foreach.hpp>

namespace sb = SageBuilder;
namespace si = SageInterface;

namespace physis {
namespace translator {
namespace rose_fortran {

SgDerivedTypeStatement *BuildDerivedTypeStatementAndDefinition(
    string name, SgScopeStatement *scope) {
  // This function builds a class declaration and definition 
  // (both the defining and nondefining declarations as required).

  // This is the class definition (the fileInfo is the position of the opening brace)
  SgClassDefinition* classDefinition   = new SgClassDefinition();
  assert(classDefinition != NULL);

  // DQ (11/28/2010): Added specification of case insensitivity for Fortran.
  classDefinition->setCaseInsensitive(true);

  // classDefinition->set_endOfConstruct(SOURCE_POSITION);
  si::setOneSourcePositionForTransformation(classDefinition);

  // Set the end of construct explictly (where not a transformation this is the location of the closing brace)
  // classDefinition->set_endOfConstruct(SOURCE_POSITION);

  // This is the defining declaration for the class (with a reference to the class definition)
  SgDerivedTypeStatement* classDeclaration = new SgDerivedTypeStatement(name.c_str(),SgClassDeclaration::e_struct,NULL,classDefinition);
  assert(classDeclaration != NULL);
  // classDeclaration->set_endOfConstruct(SOURCE_POSITION);

  si::setOneSourcePositionForTransformation(classDeclaration);

  // Set the defining declaration in the defining declaration!
  classDeclaration->set_definingDeclaration(classDeclaration);

  // Set the non defining declaration in the defining declaration (both are required)
  SgDerivedTypeStatement* nondefiningClassDeclaration = new SgDerivedTypeStatement(name.c_str(),SgClassDeclaration::e_struct,NULL,NULL);
  assert(classDeclaration != NULL);

  // DQ (12/27/2010): Set the parent before calling the SgClassType::createType() since then name mangling will require it.
  // Set the parent explicitly
  nondefiningClassDeclaration->set_parent(scope);

  // Liao 10/30/2009. we now ask for explicit creation of SgClassType. The constructor will not create it by default
  if (nondefiningClassDeclaration->get_type () == NULL) 
    nondefiningClassDeclaration->set_type (SgClassType::createType(nondefiningClassDeclaration));
  classDeclaration->set_type(nondefiningClassDeclaration->get_type());

  // nondefiningClassDeclaration->set_endOfConstruct(SOURCE_POSITION);

  // Leave the nondefining declaration without a specific source code position.
  si::setOneSourcePositionForTransformation(nondefiningClassDeclaration);

  // Set the internal reference to the non-defining declaration
  classDeclaration->set_firstNondefiningDeclaration(nondefiningClassDeclaration);

  // DQ (12/27/2010): Moved to before call to SgClassType::createType().
  // Set the parent explicitly
  // nondefiningClassDeclaration->set_parent(scope);

  // Set the defining and no-defining declarations in the non-defining class declaration!
  nondefiningClassDeclaration->set_firstNondefiningDeclaration(nondefiningClassDeclaration);
  nondefiningClassDeclaration->set_definingDeclaration(classDeclaration);

  // Set the nondefining declaration as a forward declaration!
  nondefiningClassDeclaration->setForward();

  // Don't forget the set the declaration in the definition (IR node constructors are side-effect free!)!
  classDefinition->set_declaration(classDeclaration);

  // set the scope explicitly (name qualification tricks can imply it is not always the parent IR node!)
  classDeclaration->set_scope(scope);
  nondefiningClassDeclaration->set_scope(scope);

  // Set the parent explicitly
  classDeclaration->set_parent(scope);

  // A type should have been build at this point, since we will need it later!
  ROSE_ASSERT(classDeclaration->get_type() != NULL);

  // We use the nondefiningClassDeclaration, though it might be that for Fortran the rules that cause this to be important are not so complex as for C/C++.
  SgClassSymbol* classSymbol = new SgClassSymbol(nondefiningClassDeclaration);

  // Add the symbol to the current scope (the specified input scope)
  scope->insert_symbol(name,classSymbol);

  ROSE_ASSERT(scope->lookup_class_symbol(name) != NULL);

  // some error checking
  assert(classDeclaration->get_definingDeclaration() != NULL);
  assert(classDeclaration->get_firstNondefiningDeclaration() != NULL);
  assert(classDeclaration->get_definition() != NULL);

  ROSE_ASSERT(classDeclaration->get_definition()->get_parent() != NULL);
#if 0
  // DQ (8/28/2010): Save the attributes used and clear the astAttributeSpecStack for this declaration (see test2010_34.f90).
  while (astAttributeSpecStack.empty() == false)
  {
    setDeclarationAttributeSpec(classDeclaration,astAttributeSpecStack.front());

    if (astAttributeSpecStack.front() == AttrSpec_PUBLIC || astAttributeSpecStack.front() == AttrSpec_PRIVATE)
    {
      // printf ("astNameStack.size() = %zu \n",astNameStack.size());
      if (astNameStack.empty() == false)
      {
        string type_attribute_string = astNameStack.front()->text;
        // printf ("type_attribute_string = %s \n",type_attribute_string.c_str());
        astNameStack.pop_front();
      }
    }

    astAttributeSpecStack.pop_front();
  }
#endif
  
  return classDeclaration;
}


SgFortranDo *BuildFortranDo(SgExpression *initialization,
                            SgExpression *bound,
                            SgExpression *increment,
                            SgBasicBlock *body) {
  // Based on FortranParserActionROSE.C
  // No explict increment if it's 1
#if 0 // DOES NOT WORK
  if (isSgIntVal(increment) &&
      isSgIntVal(increment)->get_value() == 1) {
    increment = NULL;
  }
#endif  
  SgFortranDo *fd = new SgFortranDo(initialization, bound,
                                    increment, body);
  fd->setCaseInsensitive(true);
  body->set_parent(fd);
  si::setOneSourcePositionForTransformation(fd);
  fd->set_old_style(false);
  fd->set_has_end_statement(true);
  return fd;
}

SgAllocateStatement *BuildAllocateStatement() {
  SgAllocateStatement *s = new SgAllocateStatement();
  si::setOneSourcePositionForTransformation(s);
  return s;
}

}  // namespace rose_fortran
}  // namespace translator
}  // namespace physis
