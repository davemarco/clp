grammar Sql;

import SqlBase;

standaloneSelectItemList
    : selectItemList EOF
    ;

standaloneRelationList
    : relationList EOF
    ;

standaloneBooleanExpression
    : booleanExpression EOF
    ;

standaloneSortItemList
    : sortItemList EOF
    ;

standaloneIntegerValue
    : INTEGER_VALUE EOF
    ;
